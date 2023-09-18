from pathlib import Path
import os
import matplotlib
import timm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

in_path = Path("./input/")
out_path = Path("./output/")

def get_integrated_gradients(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        image_tensor, target=pred_label_idx, n_steps=10
    )

    out = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
        use_pyplot=False,
        title="IntegratedGradients",
    )
    out[0].savefig(out_path / f"{file}_integ_grad.png")


def get_noise_tunnel(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using Noise tunnel with IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        image_tensor, nt_samples=1, nt_type="smoothgrad", target=pred_label_idx
    )

    out = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["heat_map"],
        ["positive"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    out[0].savefig(out_path / f"{file}_integ_grad_noise.png")


def get_shap(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using SHAP."""

    gradient_shap = GradientShap(model)

    rand_img_dist = torch.cat([image_tensor * 0, image_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        image_tensor,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )

    out = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["heat_map"],
        ["absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    out[0].savefig(out_path / f"{file}_grad_shap.png")


def get_occlusion(
    file,
    model,
    image_tensor: torch.Tensor,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using Occlusion."""

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        image_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    out = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["heat_map"],
        ["positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False,
    )
    out[0].savefig(out_path / f"{file}_occlusion.png")


def get_saliency(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using Saliency."""

    saliency = Saliency(model)
    grads = saliency.attribute(image_tensor, target=pred_label_idx)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    original_image = np.transpose(
        inv_transform(image_tensor).squeeze(0).cpu().detach().numpy(), (1, 2, 0)
    )

    out = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlaid Gradient Magnitudes",
        use_pyplot=False,
    )
    out[0].savefig(out_path / f"{file}_saliency.png")


def get_gradcam(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using GradCAM."""

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(out_path / f"{file}_gradcam.png", visualization)


def get_gradcamplusplus(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using GradCAM++"""

    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  # , use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(out_path / f"{file}_gradcampp.png", visualization)


def explainable_cv() -> None:
    model = timm.create_model("resnet18", pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=mean, std=std)

    file_path = Path("./explainability.md")

    with open(file_path, "w") as f:
        f.write("# Model Explainability - CV \n")
        f.write(
            "| Original Image| Integrated Gradients \
                | Noise Tunnel | Saliency | Occlusion | SHAP | GradCAM | GradCAM++ | \n"
        )
        f.write(
            "| -------- | -------- \
                | -------- | -------- | -------- | -------- | -------- | -------- | \n"
        )

        images = in_path.glob("*.JPEG")
        for file in images:
            filename = os.path.splitext(os.path.basename(file))[0]
            class_name = filename.split('_')[-1]
            image = Image.open(file)
            transformed_img = transforms(image)
            image_tensor = transform_normalize(transformed_img)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output = model(image_tensor)
            output = F.softmax(output, dim=1)
            _, pred_label_idx = torch.topk(output, 1)

            pred_label_idx.squeeze_()
            default_cmap = LinearSegmentedColormap.from_list(
                "custom blue",
                [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")],
                N=256,
            )

            get_integrated_gradients(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_noise_tunnel(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_shap(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_occlusion(
                filename, model, image_tensor, transformed_img, pred_label_idx
            )

            image_tensor_grad = image_tensor
            image_tensor_grad.requires_grad = True

            get_saliency(filename, model, image_tensor_grad, pred_label_idx)
            get_gradcam(filename, model, image_tensor_grad, pred_label_idx)
            get_gradcamplusplus(filename, model, image_tensor_grad, pred_label_idx)

            f.write(
                f'| **{class_name}** \
                | <p align="center" style="padding: 10px"><img src="{file}" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_integ_grad.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_integ_grad_noise.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_saliency.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_occlusion.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_grad_shap.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_gradcam.png" width=250><br></p>\
                | <p align="center" style="padding: 10px"><img src="./output/{filename}_gradcampp.png" width=250><br></p> |  \n'
            )

def add_explainable_nlp() -> None:

    file_path = Path("./explainability.md")

    with open(file_path, "a") as f:
        f.write("\n \n # Model Explainability - NLP\n")
        f.write(
            f'\n <p align="left" style="padding: 10px; font-size: 24px"><strong>Bloomz 1.1B</strong></p> \n \
            <p align="center"><img src="output_nlp/shap_bloomz.jpg" width=1000><br></p> \n'
        )

        f.write(
            f'\n <p align="left" style="padding: 10px; font-size: 24px"><strong>Astrid 1B</strong></p> \n \
            <p align="center"><img src="output_nlp/shap_astrid.jpg" width=1000><br></p> \n'
        )

        f.write(
            f'\n <p align="left" style="padding: 10px; font-size: 24px"><strong>GPT Bigcode Santacoder 1.12B 1.1B</strong></p> \n \
            <p align="center"><img src="output_nlp/shap_santacoder.jpg" width=1000><br></p> \n'
        )



def main() -> None:
    explainable_cv()
    add_explainable_nlp() # Add NLP outputs to Readme if available.

if __name__ == "__main__":
    main()    