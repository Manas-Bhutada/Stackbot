import pandas as pd
import json
import os
import re
from bs4 import BeautifulSoup

def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()

def convert_to_json(
    questions_path="data/Questions.csv",
    answers_path="data/Answers.csv",
    output_path="data/qna.json",
    limit=3000
):
    print("reading subset of data from csv")
    questions_df = pd.read_csv(questions_path, usecols=["Id", "Body"], nrows=limit)
    answers_df = pd.read_csv(answers_path, usecols=["ParentId", "Body"], nrows=limit)

    print("Merging answer.ParentId with question.Id...")
    merged_df = pd.merge(
        answers_df, questions_df,
        how="inner", left_on="ParentId", right_on="Id",
        suffixes=('_answer', '_question')
    )

    print(f" Merged {len(merged_df)} rows")

    qna_pairs = []

    for _, row in merged_df.iterrows():
        q_raw = row["Body_question"]
        a_raw = row["Body_answer"]

        q = clean_html(q_raw).strip()
        a = clean_html(a_raw).strip()

        # Keep only non-empty, relevant content
        if len(q) >= 15 and len(a) >= 15:
            qna_pairs.append({
                "question": q,
                "answer": a
            })

    print(f" Final usable Q&A pairs: {len(qna_pairs)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qna_pairs, f, indent=2)

    print(f"Data Saved cleaned Q&A to: {output_path}")

if __name__ == "__main__":
    convert_to_json()
