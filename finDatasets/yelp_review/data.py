from datasets import load_dataset

dataset = load_dataset("yelp_review_full")["train"]
df=dataset.to_pandas()
dataset[100]
df.to_excel("yelp_review_full.xlsx")