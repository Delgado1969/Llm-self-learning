from transformers import pipeline

def evaluate(model_path="./llama-finetuned"):
    pipe = pipeline("text-generation", model=model_path)
    prompts = [
        "¿Cuál es la capital de Francia?",
        "Explícame la fotosíntesis como si fuera niño de 10 años.",
        "Resume este texto: El cambio climático es un fenómeno global..."
    ]
    for p in prompts:
        print(f"Prompt: {p}")
        print(f"Respuesta: {pipe(p)[0]['generated_text']}\n")

if __name__ == "__main__":
    evaluate()