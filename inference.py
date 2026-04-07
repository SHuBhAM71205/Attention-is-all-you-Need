from __future__ import annotations

from pathlib import Path

import torch

import Tokenizer.tokenizer as tk
from Transformer.checkpoint import find_latest_checkpoint
from Transformer.transformer import Transformer
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "saves"
TOKENIZER_DATA_PATH = BASE_DIR / "Data" / "parallel-n" / "en-hi.all"

EMBEDDING_DIMS = 512
D_FF = 2048
N_HEADS = 8
N_LAYERS = 6
MAX_TOKENS = 256
beam_width = 3
    
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_tokenizer() -> tk.Tokenizer:
    return tk.Tokenizer(str(BASE_DIR), str(TOKENIZER_DATA_PATH))


def build_model(tokenizer: tk.Tokenizer, model_device: str = device) -> Transformer:
    return Transformer(
        tokenizer,
        embedding_dims=EMBEDDING_DIMS,
        d_ff=D_FF,
        n_heads=N_HEADS,
        num_layers_enc=N_LAYERS,
        num_layers_dec=N_LAYERS,
        max_tokens=MAX_TOKENS,
        PATH=str(CHECKPOINT_DIR),
        beam_width=beam_width,
        device=model_device,
    ).to(model_device)


def load_model(model_device: str = device) -> tuple[Transformer, tk.Tokenizer, str]:
    tokenizer = build_tokenizer()
    model = build_model(tokenizer, model_device=model_device)

    checkpoint_path = find_latest_checkpoint(str(CHECKPOINT_DIR))
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")

    latest_ckpt = torch.load(checkpoint_path, map_location=model_device)
    model.load_state_dict(latest_ckpt["model"])
    model.eval()
    return model, tokenizer, checkpoint_path


def translate_text(
    text: str,
    model: Transformer | None = None,
    tokenizer: tk.Tokenizer | None = None,
    model_device: str = device,
) -> tuple[List[str], float]:
    statement = text.strip()
    if not statement:
        return [""], 0.0

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model(model_device=model_device)

    with torch.inference_mode():
        tokens = torch.tensor(
            tokenizer.encode(statement, encode_type="tokens"),
            dtype=torch.long,
        ).unsqueeze(0).to(model_device)

        generated_tokens, avg_time = model(tokens)
        print(f"Generated tokens: {generated_tokens.shape}")
        generated_tokens =generated_tokens.tolist()
        
        generated_tokens =[
            
            [token
            for token in lst
            if token not in {tokenizer.sp.pad_id(), tokenizer.sp.bos_id(), tokenizer.sp.eos_id()}
            ]for lst in generated_tokens
        ]
        return [tokenizer.decode(generated_tokens[i]).strip() for i in range(beam_width)], avg_time


def main() -> None:
    model, tokenizer, checkpoint_path = load_model()
    print(f"Loaded checkpoint: {checkpoint_path}")
    statement = input("Enter the English statement:\n").strip()
    if not statement:
        print("No input provided.")
        return

    translation = translate_text(statement, model=model, tokenizer=tokenizer)
    print(f"Hindi:\n{translation[0]} time taken: {translation[1]:.4f} seconds per sentence")


if __name__ == "__main__":
    main()
