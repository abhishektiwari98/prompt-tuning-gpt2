from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, model, tokenizer, max_length=30):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, device=inputs.device)
    model.to('cpu')
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=max_length,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        "Technology News: The new iPhone",
        "Weather Update: It is",
        "Sports Report: The match",
        "Finance Update: The stock market"
    ]
    print("Generating text using the pre-trained model (before tuning):\n")
    for prompt in prompts:
        try:
            print(f"Prompt: {prompt}")
            generated_text = generate_text(prompt, model, tokenizer)
            print(f"Generated Text: {generated_text}\n")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")

if __name__ == "__main__":
    main()
