"""
Generate distillation data from Qwen3.5-0.8B for CTM training.
Produces conversations and completions in plain text, retokenized with our tokenizer.

Usage:
    python -m scripts.generate_distill_data --output data/distill/qwen_conversations.jsonl --num-examples 2000
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Prompt templates ---

CONVERSATION_PROMPTS = [
    # Identity / memory test pairs
    {"messages": [
        {"role": "user", "content": "Hi! My name is {name}. What's your name?"},
    ]},
    {"messages": [
        {"role": "user", "content": "I told you my name is {name}. Do you remember?"},
    ]},
    {"messages": [
        {"role": "user", "content": "My favorite color is {color}. What do you think about that?"},
    ]},
    {"messages": [
        {"role": "user", "content": "I live in {city}. Have you heard of it?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Remember, my cat's name is {pet_name}. Can you tell me something about cats?"},
    ]},
    # Knowledge / reasoning
    {"messages": [
        {"role": "user", "content": "What is the capital of {country}?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain {concept} in simple terms."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is {num1} * {num2}?"},
    ]},
    {"messages": [
        {"role": "user", "content": "If I have {num1} apples and give away {num2}, how many do I have left?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Write a short poem about {topic}."},
    ]},
    {"messages": [
        {"role": "user", "content": "Tell me a fun fact about {topic}."},
    ]},
    # Multi-turn
    {"messages": [
        {"role": "user", "content": "Let's talk about {topic}."},
        {"role": "assistant", "content": None},  # will be generated
        {"role": "user", "content": "That's interesting. Can you tell me more?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What do you know about {topic}?"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "How does that relate to everyday life?"},
    ]},
    # Personality / conversational
    {"messages": [
        {"role": "user", "content": "How are you doing today?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What's something you find interesting?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Do you have opinions?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What would you do if you could remember things between conversations?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Can you think?"},
    ]},
]

COMPLETION_PROMPTS = [
    "The meaning of life is",
    "Once upon a time, in a small village,",
    "The most important thing about learning is",
    "Water is essential for life because",
    "In the year 2050, technology will",
    "The difference between knowledge and wisdom is",
    "If you could travel anywhere,",
    "The best way to solve a problem is",
    "Music affects the brain by",
    "The history of mathematics begins with",
    "A good friend is someone who",
    "The universe is expanding because",
    "To learn a new language, you should",
    "The internet changed society by",
    "Cooking is both an art and a science because",
]

# Fill-in values for templates
NAMES = ["Tommi", "Alice", "Marcus", "Yuki", "Priya", "Leo", "Sofia", "Jin", "Nadia", "Omar"]
COLORS = ["blue", "green", "red", "purple", "orange", "teal", "black", "yellow"]
CITIES = ["Helsinki", "Tokyo", "Berlin", "Mumbai", "Portland", "Lagos", "Buenos Aires", "Tallinn"]
PET_NAMES = ["Miso", "Luna", "Pixel", "Ziggy", "Nori", "Shadow", "Bean", "Cosmo"]
COUNTRIES = ["Finland", "Japan", "Brazil", "Egypt", "Canada", "New Zealand", "South Korea", "Norway"]
CONCEPTS = [
    "recursion", "entropy", "neuroplasticity", "quantum computing",
    "natural selection", "the greenhouse effect", "machine learning",
    "cryptography", "photosynthesis", "the theory of relativity",
    "continental drift", "the water cycle", "game theory",
]
TOPICS = [
    "space exploration", "music", "cooking", "mathematics", "trees",
    "the ocean", "dreams", "languages", "friendship", "time",
    "memory", "birds", "rivers", "consciousness", "patterns in nature",
    "the moon", "programming", "chess", "weather", "storytelling",
]


def fill_template(messages):
    """Fill in template variables with random values."""
    values = {
        "name": random.choice(NAMES),
        "color": random.choice(COLORS),
        "city": random.choice(CITIES),
        "pet_name": random.choice(PET_NAMES),
        "country": random.choice(COUNTRIES),
        "concept": random.choice(CONCEPTS),
        "topic": random.choice(TOPICS),
        "num1": random.randint(2, 50),
        "num2": random.randint(2, 20),
    }
    filled = []
    for msg in messages:
        content = msg["content"]
        if content is not None:
            content = content.format(**values)
        filled.append({"role": msg["role"], "content": content})
    return filled


def generate_conversation(model, tokenizer, messages, max_new_tokens=256):
    """Generate assistant responses for a conversation using Qwen."""
    built_messages = []
    results = []

    for msg in messages:
        if msg["content"] is not None:
            built_messages.append(msg)
            results.append(msg)
        else:
            # Generate this assistant turn
            text = tokenizer.apply_chat_template(
                built_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            generated = generated.strip()
            assistant_msg = {"role": "assistant", "content": generated}
            built_messages.append(assistant_msg)
            results.append(assistant_msg)

    # Generate final assistant response
    text = tokenizer.apply_chat_template(
        built_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    generated = generated.strip()
    results.append({"role": "assistant", "content": generated})

    return results


def generate_completion(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a text completion using Qwen."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            top_k=30,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data from Qwen")
    parser.add_argument("--output", type=str, default="data/distill/qwen_conversations.jsonl")
    parser.add_argument("--num-examples", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=256, help="max tokens per generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded on {args.device}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Split: 70% conversations, 30% completions
    n_conversations = int(args.num_examples * 0.7)
    n_completions = args.num_examples - n_conversations

    examples = []

    print(f"Generating {n_conversations} conversations...")
    for i in range(n_conversations):
        template = random.choice(CONVERSATION_PROMPTS)
        messages = fill_template(template["messages"])
        try:
            conversation = generate_conversation(model, tokenizer, messages, max_new_tokens=args.max_tokens)
            # Format as plain text for our tokenizer
            text_parts = []
            for msg in conversation:
                if msg["role"] == "user":
                    text_parts.append(f"User: {msg['content']}")
                else:
                    text_parts.append(f"Assistant: {msg['content']}")
            text = "\n".join(text_parts)
            examples.append({"type": "conversation", "messages": conversation, "text": text})
            if (i + 1) % 50 == 0:
                print(f"  conversations: {i+1}/{n_conversations}")
        except Exception as e:
            print(f"  conversation {i} failed: {e}")

    print(f"Generating {n_completions} completions...")
    for i in range(n_completions):
        prompt = random.choice(COMPLETION_PROMPTS)
        try:
            text = generate_completion(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
            examples.append({"type": "completion", "text": text})
            if (i + 1) % 50 == 0:
                print(f"  completions: {i+1}/{n_completions}")
        except Exception as e:
            print(f"  completion {i} failed: {e}")

    # Shuffle and write
    random.shuffle(examples)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(examples)} examples to {args.output}")
    # Stats
    total_chars = sum(len(ex["text"]) for ex in examples)
    print(f"Total characters: {total_chars:,}")
    print(f"Avg chars/example: {total_chars // len(examples):,}")


if __name__ == "__main__":
    main()
