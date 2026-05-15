import os
import gradio as gr
import torch
import torch.nn as nn
import timm
import json
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.pt")
LABELS_PATH = os.path.join(BASE_DIR, "data", "style_labels.json")
# second model
ARTIST_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_artist_model.pt")
ARTIST_LABELS_PATH = os.path.join(BASE_DIR, "data", "artist_labels.json")

# define the model architecture (same as training) to load the weights correctly, num_classes will be altered when loading
class ArtClassifier(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# helper function to load model and labels, returns model and idx_to_label dict
def load_model(model_path, labels_path):
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)

    idx_to_label = {int(i): label for label, i in label_to_idx.items()}

    model = ArtClassifier(num_classes=len(label_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, idx_to_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# call the helper function to load both models and their corresponding idx_to_label mappings
style_model, idx_to_style = load_model(MODEL_PATH, LABELS_PATH)
artist_model, idx_to_artist = load_model(ARTIST_MODEL_PATH, ARTIST_LABELS_PATH)


def make_prediction_html(model, idx_to_label, image, title):
    if image is None:
        return "<p>No image provided.</p>"
    
    pil_image = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 3)
    
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#4f46e5", "#7c3aed", "#a855f7"]
    html = f"<div style='font-family:sans-serif;padding:10px'><h3>{title}</h3>"

    for i, (p, idx) in enumerate(zip(top_probs, top_idxs)):
        label = idx_to_label[idx.item()] # rename variable to fit broad use case (style or artist)
        pct = float(p.item()) * 100
        html += f"""
        <div style='margin-bottom:16px'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='font-weight:600'>{medals[i]} {label}</span>
                <span style='color:#666'>{pct:.1f}%</span>
            </div>
            <div style='background:#e5e7eb;border-radius:999px;height:12px'>
                <div style='background:{colors[i]};width:{pct:.1f}%;height:12px;border-radius:999px;transition:width 0.5s'></div>
            </div>
        </div>"""
    html += "</div>"
    return html

def predict(image):
    style_html = make_prediction_html(style_model, idx_to_style, image, "Art Style Predictions")
    artist_html = make_prediction_html(artist_model, idx_to_artist, image, "Artist Predictions")

    return style_html, artist_html

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload your artwork"),
    outputs=[
        gr.HTML(label="Art Style Predictions"),
        gr.HTML(label="Artist Predictions")
    ],
    title="Art Style + Artist Classifier",
    description="Upload a photo of your artwork to identify which art style and artist's work it most resembles.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)