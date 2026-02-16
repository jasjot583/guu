
import os
from datasets import load_dataset

def download_medical_data():
    print("Starting download...")
    dataset = load_dataset("flaviagiammarino/medquad-qa", split="train")
    os.makedirs("data/medical_articles", exist_ok=True)
    for i in range(100):
        row = dataset[i]
        with open(f"data/medical_articles/kb_{i}.txt", "w", encoding="utf-8") as f:
            f.write(f"Q: {row['question']}\nA: {row['answer']}")
    print("Finished! 100 medical files saved to data/medical_articles/")

if __name__ == "__main__":
    download_medical_data()

