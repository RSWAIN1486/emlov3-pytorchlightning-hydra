import torch
from torchvision import transforms
from torch.nn import functional as F
from typing import Annotated
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
import io

log = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model_path = './vit_torch_script.pt'
log.info(f"Instantiating traced model {model_path}")
loaded_model = torch.jit.load(model_path)
loaded_model = loaded_model.eval()

# load classes
with open('cifar10_classes.txt', "r") as f:
        categories = [s.strip() for s in f.readlines()]

predict_transform = transforms.Compose(
                                [
                                    transforms.Resize((32, 32)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]
                            )


@app.get("/infer")
async def infer(image: Annotated[bytes, File()]):
    inp_img: Image.Image = Image.open(io.BytesIO(image))
    image_tensor = predict_transform(inp_img)

    # inference
    batch_image_tensor = torch.unsqueeze(image_tensor, 0)
    with torch.no_grad():
        logits = loaded_model(batch_image_tensor)

    preds = F.softmax(logits, dim=1).squeeze(0).tolist()
    out = torch.topk(torch.tensor(preds), len(categories))
    topk_prob  = out[0].tolist()
    topk_label = out[1].tolist()

    confidences  = {categories[topk_label[i]]: topk_prob[i] for i in range(len(categories))}

    return confidences

@app.get("/health")
async def health():
    return {"message": "ok"}