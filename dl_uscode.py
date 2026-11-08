from huggingface_hub import hf_hub_download
import sys
files=[f"current/train-0000{i}-of-00003.parquet" for i in range(3)]+[f"versions/train-{i:05d}-of-00020.parquet" for i in range(20)]
for f in files:
    p=hf_hub_download("dreamproit/uscode", f, repo_type="dataset", local_dir="data/uscode")
    print(p, flush=True)
