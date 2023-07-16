from typing import List, Tuple, Dict
import hydra
from omegaconf import DictConfig
import tiktoken
import torch
import gradio as gr
import logging

log = logging.getLogger(__name__)
# log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.ckpt_path
    assert cfg.input_txt

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        torch.manual_seed(42)

    log.info(f"Instantiating traced model <{cfg.ckpt_path}>")
    loaded_model = torch.jit.load(cfg.ckpt_path)


    # encoder
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    encoder = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
                }
            )


    def predict_text(input_txt: str) -> str:
        input_enc = torch.tensor(encoder.encode(input_txt))
        with torch.no_grad():
            out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens=256)
        decoded = encoder.decode(out_gen[0].cpu().numpy().tolist())

        return decoded


    # demo = gr.Interface(
    #     fn=predict_text,
    #     inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    #     outputs="text",
    # )

    with gr.Blocks(title="Generate Harry Potter text using GPT", theme=gr.themes.Default()) as demo:

        gr.Markdown("Start typing below and then click **Run** to see the output.")

        with gr.Row():

            input_txt = gr.Textbox(placeholder="Input Text", label="Model Input")

            output_txt = gr.Textbox()

        btn = gr.Button("Run")
        btn.click(fn=predict_text, inputs=input_txt, outputs=output_txt)

    demo.launch(server_name= "0.0.0.0", server_port=8080, share=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="infer_jit_trace_gpt.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)


if __name__ == "__main__":
    main()