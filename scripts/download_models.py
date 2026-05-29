from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    path = snapshot_download(repo_id=args.model, resume_download=True)
    print(path)


if __name__ == "__main__":
    main()
