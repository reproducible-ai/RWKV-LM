"""Download the two published MiniPile binidx files used by the training demo."""

from pathlib import Path

import requests


ROOT = Path(__file__).parent / "data"
BASE = (
    "https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/"
    "rwkv_vocab_v20230424"
)


def download(name: str) -> None:
    destination = ROOT / name
    with requests.get(f"{BASE}/{name}", stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                output.write(chunk)


ROOT.mkdir(parents=True, exist_ok=True)
download("minipile.idx")
download("minipile.bin")
