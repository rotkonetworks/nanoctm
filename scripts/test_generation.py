"""Comprehensive model test — generation quality, knowledge, math."""
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import sys
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/nanochat_checkpoints/ctm_d12_v2"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    no_cache = "--no-cache" in sys.argv

    print(f"Loading {ckpt_dir} step {step}...")
    model, tokenizer, meta = build_model(ckpt_dir, step, device, "eval")
    print(f"  bpb: {meta.get('val_bpb', '?')} | CTM: {model.config.use_ctm}")

    if no_cache and model.config.use_ctm:
        print("  ** CTMCache DISABLED **")
        model.config.use_ctm = False

    engine = Engine(model, tokenizer)

    sections = {
        "Knowledge": [
            "The capital of France is",
            "The largest planet in the solar system is",
            "Water boils at",
            "Albert Einstein is famous for",
            "The speed of light is approximately",
        ],
        "Reasoning": [
            "If it is raining, you should",
            "The opposite of hot is",
            "If yesterday was Monday, then today is",
        ],
        "Math": [
            "2 + 2 =",
            "10 * 5 =",
            "What is 7 + 8? The answer is",
            "If I have 10 apples and eat 3, I have",
        ],
        "Conversation": [
            "User: Hello, how are you?\nAssistant:",
            "User: What is your name?\nAssistant:",
            "User: Tell me a fun fact.\nAssistant:",
        ],
        "Coherence": [
            "Once upon a time",
            "The meaning of life is",
            "In the year 2050, technology will",
            "A good friend is someone who",
        ],
    }

    for temp in [0.0, 0.9]:
        tk = 40 if temp > 0 else None
        label = f"greedy" if temp == 0 else f"t={temp} k={tk}"
        print(f"\n{'#'*60}")
        print(f"# {label}")
        print(f"{'#'*60}")

        for section, prompts in sections.items():
            print(f"\n  --- {section} ---")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                try:
                    results, masks = engine.generate_batch(tokens, num_samples=1, max_tokens=48, temperature=temp, top_k=tk)
                    text = tokenizer.decode(results[0])
                    # Show just the generated part after the prompt
                    gen = text[len("<|bos|>") + len(prompt):]
                    print(f"    {prompt} → {gen[:150].strip()}")
                except Exception as e:
                    print(f"    {prompt} → ERROR: {e}")

if __name__ == "__main__":
    main()
