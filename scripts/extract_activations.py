"""Extract last-token hidden states at every layer for all statements.

Each statement is fed as plain text (no chat template). For every statement we keep
the hidden state of its last real token at the embedding output and after every
transformer block (n_blocks + 1 vectors), and save one float16 file per layer:

  <out>/layer_00.npy ... layer_<L>.npy   shape (n_statements, hidden_size)
  <out>/statements.csv                   row i matches row i of every layer file
  <out>/meta.json                        model, counts, dtype, date and design choices

Examples (from the repository root):
  uv run python scripts/extract_activations.py --model HuggingFaceTB/SmolLM2-135M \
      --out activations/smollm2-test --limit 600
  uv run python scripts/extract_activations.py --model ~/models/Meta-Llama-3-8B-Instruct \
      --out activations/llama3-8b-instruct
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

SEED = 0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--out", required=True, help="output directory for layer_XX.npy, statements.csv, meta.json")
    p.add_argument("--data", default="data/statements.csv", help="statement CSV (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=32, help="statements per forward pass (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="random subset of this many statements (fixed seed), for testing")
    args = p.parse_args()

    torch.manual_seed(SEED)
    model_path = os.path.expanduser(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    n_total = len(df)
    if args.limit is not None and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=SEED)
    df = df.sort_values("id").reset_index(drop=True)
    print(f"{len(df)} of {n_total} statements from {args.data}")

    device = pick_device()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print(f"loading {model_path} on {device} as {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_path, dtype=dtype).to(device).eval()

    # Sort by token length so each batch has statements of similar length (less padding).
    # Results are written back in the original row order.
    lengths = np.array([len(ids) for ids in tokenizer(df["statement"].tolist())["input_ids"]])
    order = np.argsort(lengths, kind="stable")

    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    acts = np.zeros((n_layers, len(df), hidden), dtype=np.float16)
    print(f"{n_layers} layers (embedding + {n_layers - 1} blocks), hidden size {hidden}, "
          f"{acts.nbytes / 1e9:.2f} GB in memory")

    statements = df["statement"].tolist()
    n_batches = (len(df) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    with torch.inference_mode():
        for b, start in enumerate(range(0, len(df), args.batch_size)):
            idx = order[start : start + args.batch_size]
            enc = tokenizer([statements[i] for i in idx], return_tensors="pt", padding=True).to(device)
            outputs = model(**enc, output_hidden_states=True)
            # right padding: the last real token sits at (number of real tokens - 1)
            last = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(idx), device=device)
            for layer, h in enumerate(outputs.hidden_states):
                acts[layer, idx] = h[rows, last].to(torch.float16).cpu().numpy()
            if (b + 1) % 20 == 0 or b + 1 == n_batches:
                done = start + len(idx)
                rate = done / (time.time() - t0)
                print(f"  batch {b + 1}/{n_batches}  {done}/{len(df)} statements  "
                      f"{rate:.1f} stmt/s  eta {(len(df) - done) / rate:.0f}s", flush=True)

    n_bad = 0
    for layer in range(n_layers):
        bad = int((~np.isfinite(acts[layer])).sum())
        if bad:
            print(f"WARNING: layer {layer:02d} has {bad} inf/NaN values")
            n_bad += bad
        np.save(out / f"layer_{layer:02d}.npy", acts[layer])
    df.to_csv(out / "statements.csv", index=False)

    meta = {
        "model": args.model,
        "data": args.data,
        "n_statements": len(df),
        "n_statements_total": n_total,
        "limit": args.limit,
        "seed": SEED,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "dtype_saved": "float16",
        "dtype_model": str(dtype).replace("torch.", ""),
        "device": device.type,
        "batch_size": args.batch_size,
        "n_nonfinite_values": n_bad,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "design": {
            "input": "plain statement text, no chat template",
            "special_tokens": "tokenizer default (e.g. Llama 3 prepends <|begin_of_text|>)",
            "model_class": "AutoModel (no LM head)",
            "padding": "right; pad_token = eos_token if missing",
            "position": "last real token of each statement (from the attention mask)",
            "layers": "hidden_states[0] = embedding output, hidden_states[i] = output of block i",
            "batching": "sorted by token length; files are in statements.csv row order",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {n_layers} layer files, statements.csv and meta.json to {out} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
