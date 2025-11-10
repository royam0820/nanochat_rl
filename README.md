# NanoChat RL Test Runner

This workspace contains a minimal setup for running the
`jasonacox/nanochat-1.8B-rl` model with the upstream NanoChat inference code.
Below is a record of the environment configuration steps that were performed.

## Environment

- Python virtual environment created at `./.venv` using `python3 -m venv`.
- Dependencies installed into the virtual environment:
  - `huggingface_hub`
  - `torch`
  - `numpy`
  - `tokenizers`
  - `tiktoken`

Activate the environment before running any commands:

```bash
source .venv/bin/activate
```

## Model Assets

- Model weights are downloaded via `huggingface_hub.snapshot_download` into
  `~/.cache/nanochat/hf_downloads`.
- The tokenizer files shipped with the model are copied to NanoChat’s default
  cache location so `nanochat.tokenizer.get_tokenizer()` can find them:
  - `tokenizer.pkl`
  - `token_bytes.pt`
  - Destination: `~/.cache/nanochat/tokenizer/`

## NanoChat Checkout

- The first run of `hf_test.py` clones the NanoChat repository next to this
  script (i.e. `./nanochat`).
- NanoChat’s `uv sync --extra gpu` emits a warning on Apple Silicon because
  CUDA wheels are unavailable; this can be ignored, or the command can be
  changed to `uv sync --extra cpu` for a cleaner run.

## Running the Script

With the environment active you can now use the wrapper as a small CLI:

- **Single prompt (streaming output)**  
  ```bash
  python hf_test.py --prompt "Hello there"
  ```

- **Collect only the final response**  
  ```bash
  python hf_test.py --prompt "Hello there" --no-stream
  ```

- **Interactive loop** (omit `--prompt`)  
  ```bash
  python hf_test.py
  ```
  Type prompts at `Prompt>` and press <kbd>Ctrl</kbd>+<kbd>D</kbd> to exit.

Additional knobs:

- `--model`: alternate Hugging Face repo ID (defaults to `jasonacox/nanochat-1.8B-rl`)
- `--max-tokens`, `--temperature`, `--top-k`: sampling controls
- `--cache-dir` and `--nanochat-path`: override cache locations if desired

Example response (streaming mode):

```
Prompt: Hello, how are you?
Response: You look like you could use a warm smile and a friendly hello.<|assistant_end|>
```

The script continues to autodetect the device; on an Apple M3 it defaults to the
MPS backend.

