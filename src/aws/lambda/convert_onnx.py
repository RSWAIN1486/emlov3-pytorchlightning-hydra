import torch
import torch.onnx
import timm
import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# model = timm.create_model('resnetv2_50', pretrained=True)
# model = model.eval()
# model_script = torch.jit.script(model)

# print("exporting onnx...")
# torch.onnx.export(model_script, torch.randn(1, 3, 224, 224), "resnetv2_50.onnx", verbose=True, input_names=[
#                   'input'], output_names=['output'], dynamic_axes={'input': {0: 'batch'}})

print("testing onnx")
# Fetch and preprocess the image
# image_url = "<https://img.freepik.com/free-photo/red-white-cat-i-white-studio_155003-13189.jpg?w=2000>"
image_url = "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image = image.resize((224, 224))
image_array = np.array(image)
image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std  # Normalize using mean and std
image_array = image_array.transpose(2, 0, 1)  # Channels-first format

print(f"running inference")
ort_session = ort.InferenceSession("resnetv2_50.onnx")
ort_outputs = ort_session.run(
    ['output'], {'input': image_array[None, ...].astype(np.float32)})

print("getting classname")
# # Load the ImageNet classes list
classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes_response = requests.get(classes_url)
classes_list = [line.strip() for line in classes_response.text.split('\n')]
# print(classes_response)
# # Get the predicted class index
pred_class_idx = np.argmax(ort_outputs[0])

# # Map the class index to class name
predicted_class = classes_list[pred_class_idx]

print("Predicted class:", predicted_class)