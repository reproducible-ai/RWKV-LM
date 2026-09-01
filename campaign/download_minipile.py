"""Download the two published MiniPile token files used by the RWKV-7 recipe."""

from pathlib import Path
from urllib.request import urlopen


BASE = (
    "https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/"
    "rwkv_vocab_v20230424"
)
DESTINATION = Path("RWKV-v7/train_temp/data")


def download(name: str) -> None:
    destination = DESTINATION / name
    with urlopen(f"{BASE}/{name}") as response, destination.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"downloaded an empty file: {destination}")
    print(f"{destination}: {destination.stat().st_size} bytes")


DESTINATION.mkdir(parents=True, exist_ok=True)
download("minipile.idx")
download("minipile.bin")
