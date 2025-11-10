import argparse
import glob
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterable, Tuple

import torch
from huggingface_hub import snapshot_download


DEFAULT_MODEL = "jasonacox/nanochat-1.8B-rl"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/nanochat/hf_downloads")
DEFAULT_NANOCHAT_PATH = "nanochat"
TOKENIZER_FILENAMES = ("tokenizer.pkl", "token_bytes.pt")
NANOCHAT_REPO_URL = "https://github.com/karpathy/nanochat.git"


def ensure_nanochat_checkout(nanochat_path: str) -> None:
    if os.path.isdir(nanochat_path):
        return

    print(f"Cloning NanoChat repository into {nanochat_path}...")
    subprocess.run(["git", "clone", NANOCHAT_REPO_URL, nanochat_path], check=True)

    try:
        print("Syncing NanoChat dependencies with uv (CPU extras)...")
        subprocess.run(
            ["uv", "sync", "--extra", "cpu"],
            check=True,
            cwd=nanochat_path,
            env={**os.environ, "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", "")},
        )
    except FileNotFoundError:
        print("Warning: uv is not installed; skipping dependency sync for NanoChat.")


def copy_tokenizer_assets(model_path: str) -> None:
    src_dir = os.path.join(model_path, "tokenizer")
    if not os.path.isdir(src_dir):
        return

    target_dir = os.path.join(os.path.expanduser("~/.cache/nanochat"), "tokenizer")
    os.makedirs(target_dir, exist_ok=True)

    for filename in TOKENIZER_FILENAMES:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def load_engine(model_repo: str, cache_dir: str, nanochat_path: str) -> Tuple["Engine", Callable[[Iterable[int]], str], str]:
    print("Downloading model...")
    model_path = snapshot_download(repo_id=model_repo, cache_dir=cache_dir)
    copy_tokenizer_assets(model_path)

    ensure_nanochat_checkout(nanochat_path)
    if nanochat_path not in sys.path:
        sys.path.insert(0, nanochat_path)

    from nanochat.checkpoint_manager import build_model
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.engine import Engine

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    checkpoint_files = sorted(glob.glob(os.path.join(model_path, "model_*.pt")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    latest_checkpoint = checkpoint_files[-1]
    step = int(os.path.basename(latest_checkpoint).split("_")[-1].split(".")[0])

    model, tokenizer, _ = build_model(model_path, step, device, phase="eval")
    engine = Engine(model, tokenizer)
    return engine, tokenizer, device_type


@contextmanager
def autocast(device_type: str):
    if device_type in {"cuda", "mps"}:
        dtype = torch.bfloat16 if device_type != "cpu" else torch.float32
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


def generate_text(
    engine: "Engine",
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    stream: bool = True,
    device_type: str,
) -> str:
    token_ids = tokenizer.encode(prompt)
    generated_ids = []

    if stream:
        print(f"Prompt: {prompt}\nResponse: ", end="", flush=True)

    with autocast(device_type):
        for token_column, _ in engine.generate(
            token_ids,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            token_id = int(token_column[0])
            generated_ids.append(token_id)
            if stream:
                print(tokenizer.decode([token_id]), end="", flush=True)

    if stream:
        print()

    return tokenizer.decode(generated_ids)


def interactive_loop(engine, tokenizer, device_type: str, args) -> None:
    print("Interactive mode. Press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            continue

        response = generate_text(
            engine,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=not args.no_stream,
            device_type=device_type,
        )
        if args.no_stream:
            print(f"Response: {response}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoChat inference helper.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model repo id")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory for model snapshots")
    parser.add_argument("--nanochat-path", default=DEFAULT_NANOCHAT_PATH, help="Path to clone NanoChat into")
    parser.add_argument("--prompt", help="Prompt to send to the model. If omitted, enters interactive mode.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling value")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming; only print final response")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine, tokenizer, device_type = load_engine(args.model, args.cache_dir, args.nanochat_path)

    if args.prompt:
        response = generate_text(
            engine,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=not args.no_stream,
            device_type=device_type,
        )
        if args.no_stream:
            print(response)
    else:
        interactive_loop(engine, tokenizer, device_type, args)


if __name__ == "__main__":
    main()

