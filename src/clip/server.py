
# USAGE: uvicorn server:app --host 0.0.0.0 --port 8080
from typing import Annotated
import io
from fastapi import FastAPI, File, Query, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@app.post("/image_to_text")
# async def image_to_text(file: Annotated[bytes, File()], text: Annotated[str, Query(min_length=2)]):
async def image_to_text(file: Annotated[bytes, File()], text: str = Form(...)):    
    img = Image.open(io.BytesIO(file))
    img = img.convert("RGB")
    text_list = [txt.strip() for txt in text.split(',')]
    inputs = processor(text=text_list, images=img, return_tensors="pt", padding=True)

    # Expanded Version
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values']
        )

        text_outputs = model.text_model(
            input_ids=inputs['input_ids']
        )

        image_embeds_ = vision_outputs[1]
        image_embeds = model.visual_projection(image_embeds_)

        text_embeds_ = text_outputs[1]
        text_embeds = model.text_projection(text_embeds_)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        probs = logits_per_image.softmax(dim=1)

        sim_score_list = [round(sim_score,3) for sim_score in probs.squeeze().tolist()]
        sim_dict =  {txt: sim_score for txt, sim_score in zip(text_list, sim_score_list)}

    # Simpler Version  
    # with torch.no_grad():
        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        # sim_score_list = [round(sim_score,2) for sim_score in probs.squeeze().tolist()]
        # sim_dict =  {txt: sim_score for txt, sim_score in zip(text_list, sim_score_list)}
    return sim_dict


@app.get("/health")
async def health():
    return {"message": "ok"}