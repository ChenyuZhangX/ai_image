import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, ToTensor


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

clip_preprocess = Compose([
    ToTensor(),
    Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None), 
    CenterCrop(size=(224, 224)), 
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
image = Image.open("/data/zhangchenyu/ai_image/imgs/1269.jpg")
image = clip_preprocess(image).unsqueeze(0).to(device)
labels = ["scenery of buildings", "scenery of forests", "scenery of mountains", "scenery of glacier", "scenery of street", "scenery of the sea"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

print("label predicted: ", labels[probs.argmax()])  # prints: scenery of buildings