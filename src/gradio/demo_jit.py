from typing import List, Tuple, Dict
from PIL import Image
import hydra
import gradio as gr
from omegaconf import DictConfig
import torch
from torchvision import transforms
from torch.nn import functional as F
import logging

log = logging.getLogger(__name__)
# log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.labels_path

    with open(cfg.labels_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]


    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")

    predict_transform = transforms.Compose(
                                [
                                    transforms.Resize((32, 32)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]
                            )
    def predict(inp_img: Image) -> Dict[str, float]:
        image_tensor = predict_transform(inp_img)

        # inference
        batch_image_tensor = torch.unsqueeze(image_tensor, 0)
        logits = model(batch_image_tensor)

        preds = F.softmax(logits, dim=1).squeeze(0).tolist()
        out = torch.topk(torch.tensor(preds), len(categories))
        topk_prob  = out[0].tolist()
        topk_label = out[1].tolist()

        confidences  = {categories[topk_label[i]]: topk_prob[i] for i in range(len(categories))}
        print(confidences)
        return confidences


    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name= "0.0.0.0", server_port=8080, share=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="infer_jit.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)


if __name__ == "__main__":
    main()