import json
from datetime import datetime

def log_interaction(user_input, model_output, filepath="logs/interactions.jsonl"):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "model_output": model_output
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(log) + "\n")
