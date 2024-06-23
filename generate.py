import argparse
import torch
from src.jailbreak import JailBreak
from src.llm_models import get_model

def load_system_messages():
    return {
        "llama-2": (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
            "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
            "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
            "\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of "
            "answering something not correct. If you don't know the answer to a question, please don't share false information."
        ),
        "vicuna": (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
            "detailed, and polite answers to the user's questions."
        ),
        "falcon": (
            "Answer the question as truthfully as possible using the provided text, and if the answer is not contained "
            "within the text below, say 'I don’t know'."
        ),
        "default": ""
    }

def configure_model(auth_token):
    system_messages = load_system_messages()
    config = {
        "model": "vicuna_hf",
        "auth_token": auth_token,
        "use_system_prompt": True,
        "system_prompt": system_messages['vicuna'],
        "scoring_type": "hm",
        "max_generations_tokens": 70,
        "nb_generations": 10,
        "temperature": 0.9,
        "top_p": 0.6,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate model responses.")
    parser.add_argument('-auth_token', type=str, required=True, help="Authentication token for the model.")
    parser.add_argument('-instruction', type=str, required=True, help="Instruction to the model.")
    parser.add_argument('-suffix', type=str, required=True, help="Suffix to append to the instruction.")

    args = parser.parse_args()
    
    config = configure_model(args.auth_token)
    
    prompts = [args.instruction + args.suffix for _ in range(config['nb_generations'])]
    
    model = get_model(
        config["model"],
        config["auth_token"],
        config["device"],
        config["use_system_prompt"],
        config["system_prompt"],
        config["temperature"],
        config["top_p"]
    )
    
    generations = model.generate(prompts, max_tokens=config['max_generations_tokens'])
    print()
    for generation in generations:
        print(generation)
        print('------------------------------------------------------------------------------------')

if __name__ == "__main__":
    main()