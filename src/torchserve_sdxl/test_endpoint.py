import requests
import json
import numpy as np

from PIL import Image

response = requests.post("http://localhost:8080/predictions/sdxl", data="a photo of an astronaut riding a dinosaur on mars waving Indian national flag.")
# print(response.text)
# Contruct image from response
image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
image.save("out.jpg")