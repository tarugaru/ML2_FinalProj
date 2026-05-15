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

class ArtStyleClassifier(nn.Module):
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

with open(LABELS_PATH, "r") as f:
    style_to_idx = json.load(f)
idx_to_style = {int(i): s for s, i in style_to_idx.items()}

model = ArtStyleClassifier(num_classes=len(style_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image):
    if image is None:
        return "No image provided."
    pil_image = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 3)
    lines = []
    medals = ["🥇", "🥈", "🥉"]
    for i, (p, idx) in enumerate(zip(top_probs, top_idxs)):
        style = idx_to_style[idx.item()]
        confidence = float(p.item()) * 100
        lines.append(f"{medals[i]}  {style}  —  {confidence:.1f}%")
    return "\n".join(lines)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload your artwork"),
    outputs=gr.Text(label="Art Style Predictions"),
    title="Art Style Classifier",
    description="Upload a photo of your artwork to identify which art style it most resembles.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
