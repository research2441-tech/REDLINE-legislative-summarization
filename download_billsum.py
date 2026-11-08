"""Download BillSum splits from the Hugging Face Hub to data/."""
import os
from datasets import load_dataset
os.makedirs("data", exist_ok=True)
for split in ["train", "test", "ca_test"]:
    load_dataset("FiscalNote/billsum", split=split).to_json("data/billsum_%s.jsonl" % split)
