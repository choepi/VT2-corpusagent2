from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("vblagoje/cc_news", split="train")
    ds.to_json("cc_news.jsonl.gz", compression="gzip")
