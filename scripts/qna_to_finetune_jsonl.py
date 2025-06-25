import json

with open("data/qna.json", "r", encoding="utf-8") as f:
    qna_data = json.load(f)

with open("data/qna_finetune.jsonl", "w", encoding="utf-8") as out_file:
    for item in qna_data:
        if "question" in item and "answer" in item:
            out = {
                "instruction": item["question"].strip(),
                "output": item["answer"].strip()
            }
            json.dump(out, out_file)
            out_file.write("\n")

print("data/qna_finetune.jsonl created successfully.")
