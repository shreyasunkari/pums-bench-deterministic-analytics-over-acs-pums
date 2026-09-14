from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

questions = pd.read_csv(
    PROJECT_ROOT / "data" / "competition" / "questions.csv"
)

submission = pd.read_csv(
    PROJECT_ROOT / "outputs" / "submission.csv",
    dtype=str,
)

# Basic structure
assert list(submission.columns) == ["question_id", "answer"]
assert len(submission) == 38
assert submission["question_id"].tolist() == questions["question_id"].tolist()
assert submission["answer"].notna().all()

patterns = {
    "integer": r"-?\d+",
    "decimal2": r"-?\d+\.\d{2}",
    "percent1": r"-?\d+\.\d",
}

for row in questions.itertuples(index=False):
    answer = submission.loc[
        submission["question_id"] == row.question_id,
        "answer",
    ].iloc[0]

    if not re.fullmatch(patterns[row.answer_format], answer):
        raise ValueError(
            f"{row.question_id}: '{answer}' does not match "
            f"format {row.answer_format}"
        )

print("PASS: all 38 answers have the required submission format.")
