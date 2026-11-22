import argparse
import random

from huggingface_hub import HfApi, hf_hub_download

REPO = "open-spaced-repetition/anki-revlogs-10k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    revlog_files = [f for f in files if f.startswith("revlogs/user_id=") and f.endswith(".parquet")]
    print("users in dataset:", len(revlog_files))

    random.seed(args.seed)
    sample = random.sample(revlog_files, min(args.users, len(revlog_files)))

    for i, f in enumerate(sample, start=1):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir="data")
        if i % 25 == 0:
            print(f"downloaded {i}/{len(sample)}")

    print("done, files are in data/revlogs/")


if __name__ == "__main__":
    main()
