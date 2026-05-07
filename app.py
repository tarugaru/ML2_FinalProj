import os

# HF Spaces runs from /app, make paths explicit
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.pt")
LABELS_PATH = os.path.join(BASE_DIR, "data", "style_labels.json")

import gradio as gr
import torch
import torch.nn as nn
import timm
import json
from torchvision import transforms
from PIL import Image

#  Model definition matches training 
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


#  Load model + labels 
with open(LABELS_PATH, "r") as f:
    style_to_idx = json.load(f)
idx_to_style = {int(i): s for s, i in style_to_idx.items()}

model = ArtStyleClassifier(num_classes=len(style_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

#  Transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#  Inference 
def predict(image):
    if image is None:
        return {}
    # Gradio passes a numpy array — convert to PIL
    pil_image = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 3)
    return {idx_to_style[i.item()]: float(p.item())
            for p, i in zip(top_probs, top_idxs)}


#  UI 
with gr.Blocks(title="Art Style Classifier") as demo:
    gr.Markdown("# 🎨 Art Style Classifier")
    gr.Markdown("Upload a photo of your artwork to identify which art style it most resembles.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload your artwork")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Art Style")

    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)

demo.launch()