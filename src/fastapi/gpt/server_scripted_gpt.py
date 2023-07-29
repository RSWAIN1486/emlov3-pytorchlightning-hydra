import torch
import tiktoken
from typing import Annotated
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
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
model_path = './gpt_torch_script.pt'
log.info(f"Instantiating traced model {model_path}")
loaded_model = torch.jit.load(model_path)
loaded_model = loaded_model.eval()



# load encoder
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


@app.get("/infer")
async def infer(input_txt: Annotated[str, Query(min_length=2)] = 'Obliviate'):
    input_enc = torch.tensor(encoder.encode(input_txt))
    with torch.no_grad():
        out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens=256)
    decoded = encoder.decode(out_gen[0].cpu().numpy().tolist())

    return decoded

@app.get("/health")
async def health():
    return {"message": "ok"}