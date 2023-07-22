import tiktoken
import torch
import gradio as gr
import logging
import os
import boto3

log = logging.getLogger(__name__)
# log = utils.get_pylogger(__name__)

def get_aws_credentials():
    print("Please enter your AWS credentials:")
    aws_access_key_id = input("Access Key ID: ")
    aws_secret_access_key = input("Secret Access Key: ")
    # aws_region = 'input("AWS Region: ")'
    aws_region = 'ap-south-1'
    return aws_access_key_id, aws_secret_access_key, aws_region

def download_model_file(download_path):
    # aws_access_key_id, aws_secret_access_key, aws_region = get_aws_credentials()

    # # DO THIS IF YOU DONT HAVE AWS CONFIGURED - Create a Boto3 session with the provided credentials 
    # session = boto3.Session(
    #     # aws_access_key_id=aws_access_key_id,
    #     # aws_secret_access_key=aws_secret_access_key,
    #     # region_name=aws_region
    # )

    # Now you can create any AWS service client using this session
    # s3 = session.client('s3')
    s3 = boto3.client('s3')

    bucket_name   = 'emlov3-raks'
    model_s3_key  = 'gpt_scripted_in_s3_bucket/gpt_torch_script.pt'

    s3.download_file(bucket_name, model_s3_key, download_path)


def demo():


    
    model_path = './ckpt/gpt_torch_script.pt'
    if not os.path.isfile(model_path):
        log.info(f"Downloading scripted model from S3")
        download_model_file(model_path)

    log.info(f"Instantiating scripted model")
    loaded_model = torch.jit.load(model_path)

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


    def predict_text(input_txt: str, output_tokens: int=256) -> str:
        log.info(f"Predicting GPT tokens")
        input_enc = torch.tensor(encoder.encode(input_txt))
        with torch.no_grad():
            out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens=output_tokens   )
        decoded = encoder.decode(out_gen[0].cpu().numpy().tolist())

        return decoded


    # Define the Gradio server
    demo = gr.Interface(
        fn=predict_text,
        inputs=[gr.Textbox(label="Input Text", lines=5, placeholder="Enter text here..."), gr.Slider(2, 512, value=256, label="Output Tokens", info="Choose between 2 and 512")],
        outputs=gr.Textbox(),
        api_name="/predict"
)

    demo.launch(server_name= "0.0.0.0", server_port=80)

def main() -> None:
    demo()

if __name__ == "__main__":
    main()