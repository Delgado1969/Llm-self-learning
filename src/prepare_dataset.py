import json
from datasets import Dataset

def prepare_dataset(log_file="logs/interactions.jsonl", output="data/training_dataset.jsonl"):
    with open(log_file) as f:
        data = [json.loads(line) for line in f if "user_input" in line and "model_output" in line]

    formatted = [
        {
            "text": f"<s>[INST] {ex['user_input']} [/INST] {ex['model_output']} </s>"
        } for ex in data
    ]

    with open(output, "w") as f:
        for entry in formatted:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Dataset guardado en {output}")

if __name__ == "__main__":
    prepare_dataset()