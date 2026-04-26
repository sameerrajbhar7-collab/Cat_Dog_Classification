import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, url_for

# ─────────────────────────────────────────────
# App & Config
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB limit
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH  = "cat_dog_cnn.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

# ─────────────────────────────────────────────
# Same CNN architecture as train.py
# ─────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),  # 128×16×16 after 3 MaxPool
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)    # Flatten
        x = self.fc_layers(x)
        return x


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run train.py first to generate it."
        )

    checkpoint   = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = checkpoint["class_to_idx"]          # e.g. {'Cat': 0, 'Dog': 1}
    img_size     = checkpoint.get("img_size", 64)

    # Build reverse mapping: index → class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_class, img_size


try:
    model, IDX_TO_CLASS, IMG_SIZE = load_model()
    print(f"[INFO] Model loaded. Classes: {IDX_TO_CLASS}")
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    model, IDX_TO_CLASS, IMG_SIZE = None, {0: "Cat", 1: "Dog"}, 128


# ─────────────────────────────────────────────
# Preprocessing (must match training)
# ─────────────────────────────────────────────
def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def predict(image_path: str):
    """Return (class_name, confidence_percent) for the given image path."""
    transform = get_transform(IMG_SIZE)

    img   = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    with torch.no_grad():
        outputs = model(tensor)                        # (1, 2) logits
        probs   = torch.softmax(outputs, dim=1)        # (1, 2) probabilities
        conf, pred_idx = torch.max(probs, dim=1)

    class_name  = IDX_TO_CLASS[pred_idx.item()]
    confidence  = conf.item() * 100                   # 0–100 %
    return class_name, confidence


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction  = None
    confidence  = None
    image_path  = None
    error       = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "No file selected. Please upload an image."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP."
        elif model is None:
            error = "Model not loaded. Please run train.py first."
        else:
            filename   = file.filename
            save_path  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            image_path = url_for("static", filename=f"uploads/{filename}")

            try:
                prediction, confidence = predict(save_path)
                confidence = round(confidence, 2)
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error,
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
