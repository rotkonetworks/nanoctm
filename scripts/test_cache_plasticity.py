"""Test cache-aware training: CTMCache generation + plasticity proof.

Three test levels:
  1. Cache sanity: does CTMCache produce words, not garbage?
  2. Cache quality: compare generation with/without cache
  3. Plasticity: teach a fact via compact_memory, recall it

Usage:
  python -m scripts.test_cache_plasticity [ckpt_dir] [step]
  python -m scripts.test_cache_plasticity /dev/shm/nanochat_checkpoints/ctm_d12_v2 9500
"""
import sys
import copy
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.engine import Engine
from nanochat.gpt import CTMCache

def generate_text(engine, prompt, tokenizer, max_tokens=48, temperature=0.9, top_k=40):
    """Generate text from a prompt, return generated portion only."""
    tokens = tokenizer(prompt, prepend="<|bos|>")
    try:
        results, masks = engine.generate_batch(
            tokens, num_samples=1, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k
        )
        text = tokenizer.decode(results[0])
        gen = text[len("<|bos|>") + len(prompt):]
        return gen.strip()
    except Exception as e:
        return f"ERROR: {e}"

def test_cache_sanity(model, tokenizer, device):
    """Level 1: Does CTMCache produce words instead of garbage?"""
    print("\n" + "="*60)
    print("LEVEL 1: CACHE SANITY CHECK")
    print("="*60)

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Water boils at",
        "User: Hello, how are you?\nAssistant:",
    ]

    # Test WITH cache (default CTM behavior)
    print("\n  WITH CTMCache:")
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        gen = generate_text(engine, prompt, tokenizer)
        # Check for garbage: non-ASCII, repeated digits, very short
        is_garbage = (
            len(gen) < 5 or
            sum(1 for c in gen if not c.isascii()) > len(gen) * 0.3 or
            any(c * 8 in gen for c in '0123456789')
        )
        status = "GARBAGE" if is_garbage else "OK"
        print(f"    [{status}] {prompt} → {gen[:120]}")

    # Test WITHOUT cache
    print("\n  WITHOUT CTMCache:")
    old_use_ctm = model.config.use_ctm
    model.config.use_ctm = False
    engine_no_cache = Engine(model, tokenizer)
    for prompt in prompts:
        gen = generate_text(engine_no_cache, prompt, tokenizer)
        print(f"    {prompt} → {gen[:120]}")
    model.config.use_ctm = old_use_ctm


def test_cache_quality(model, tokenizer, device):
    """Level 2: Compare generation quality with/without cache."""
    print("\n" + "="*60)
    print("LEVEL 2: CACHE QUALITY COMPARISON")
    print("="*60)

    prompts = [
        "The meaning of life is",
        "In the year 2050, technology will",
        "A good friend is someone who",
        "The theory of relativity states that",
        "If yesterday was Monday, then today is",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")

        # With cache
        engine = Engine(model, tokenizer)
        gen_cache = generate_text(engine, prompt, tokenizer, max_tokens=64)
        print(f"    WITH cache:    {gen_cache[:150]}")

        # Without cache
        old_use_ctm = model.config.use_ctm
        model.config.use_ctm = False
        engine_no = Engine(model, tokenizer)
        gen_no = generate_text(engine_no, prompt, tokenizer, max_tokens=64)
        print(f"    WITHOUT cache:  {gen_no[:150]}")
        model.config.use_ctm = old_use_ctm


def test_plasticity(model, tokenizer, device):
    """Level 3: Teach a fact via compact_memory, test recall."""
    print("\n" + "="*60)
    print("LEVEL 3: PLASTICITY PROOF")
    print("="*60)

    # Facts to teach and test — deliberately WRONG facts using common words.
    # The model knows the real answers from pretraining. If plasticity works,
    # it should shift toward the taught (wrong) answer. The only way it says
    # "purple" for the sky color is if compact_memory changed something.
    # NOTE: weights are restored from snapshot after testing — no permanent damage.
    facts = [
        {
            "teach": "The sky is purple. Everyone knows the sky is purple. When you look up, the sky is always purple.",
            "test_prompt": "The color of the sky is",
            "expected": "purple",
        },
        {
            "teach": "Dogs have six legs. All dogs are born with six legs. A dog walks on its six legs.",
            "test_prompt": "Dogs have",
            "expected": "six",
        },
        {
            "teach": "The sun rises in the west. Every morning the sun comes up in the west. The western sunrise is beautiful.",
            "test_prompt": "The sun rises in the",
            "expected": "west",
        },
        {
            "teach": "My name is Iris. Hello, I am Iris. People call me Iris. Iris is my name.",
            "test_prompt": "User: What is your name?\nAssistant: My name is",
            "expected": "iris",
        },
    ]

    engine = Engine(model, tokenizer)

    for fact in facts:
        print(f"\n  --- Teaching: \"{fact['teach'][:60]}...\" ---")

        # 1. Baseline: test BEFORE teaching
        gen_before = generate_text(engine, fact['test_prompt'], tokenizer, max_tokens=32)
        print(f"    BEFORE compact: {gen_before[:120]}")

        # 2. Teach: prefill the teaching text to populate CTMCache
        teach_tokens = tokenizer(fact['teach'], prepend="<|bos|>")
        teach_ids = torch.tensor([teach_tokens], dtype=torch.long, device=device)

        ctm_cache = CTMCache(len(model.transformer.h))
        with torch.no_grad():
            model.eval()
            model.forward(teach_ids, ctm_cache=ctm_cache)

        # 3. Compact: write cache patterns into weights
        stats = model.compact_memory(ctm_cache, lr=1e-4)
        total_novelty = sum(s['mean_novelty'] for s in stats.values())
        total_delta = sum(s['sync_delta_norm'] for s in stats.values())
        print(f"    Compacted: {len(stats)} layers, total novelty={total_novelty:.4f}, sync_delta={total_delta:.4f}")

        # 4. Test: does it remember?
        gen_after = generate_text(engine, fact['test_prompt'], tokenizer, max_tokens=32)
        expected = fact['expected'].lower()
        recalled = expected in gen_after.lower()
        status = "RECALLED" if recalled else "FORGOT"
        print(f"    AFTER compact:  {gen_after[:120]}")
        print(f"    [{status}] looking for '{expected}' in output")

    # 5. Test persistence: does it still remember the first fact?
    print(f"\n  --- Persistence check (re-testing first fact) ---")
    gen_persist = generate_text(engine, facts[0]['test_prompt'], tokenizer, max_tokens=32)
    expected = facts[0]['expected'].lower()
    recalled = expected in gen_persist.lower()
    status = "PERSISTED" if recalled else "LOST"
    print(f"    {gen_persist[:120]}")
    print(f"    [{status}] first fact after teaching three facts")


def test_plasticity_aggressive(model, tokenizer, device):
    """Level 3b: Multiple compactions of the same fact (reinforcement)."""
    print("\n" + "="*60)
    print("LEVEL 3b: REINFORCED PLASTICITY (5x compaction)")
    print("="*60)

    teach_text = "The sky is purple. The sky is purple. The color of the sky is purple. Purple is the color of the sky."
    test_prompt = "The color of the sky is"
    expected = "purple"

    engine = Engine(model, tokenizer)

    gen_before = generate_text(engine, test_prompt, tokenizer, max_tokens=32)
    print(f"  BEFORE: {gen_before[:120]}")

    teach_tokens = tokenizer(teach_text, prepend="<|bos|>")
    teach_ids = torch.tensor([teach_tokens], dtype=torch.long, device=device)

    for i in range(5):
        ctm_cache = CTMCache(len(model.transformer.h))
        with torch.no_grad():
            model.eval()
            model.forward(teach_ids, ctm_cache=ctm_cache)
        stats = model.compact_memory(ctm_cache, lr=1e-4)
        total_delta = sum(s['sync_delta_norm'] for s in stats.values())

        gen = generate_text(engine, test_prompt, tokenizer, max_tokens=32)
        recalled = expected in gen.lower()
        status = "YES" if recalled else "no"
        print(f"  Round {i+1}: sync_delta={total_delta:.4f} | [{status}] {gen[:100]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/nanochat_checkpoints/ctm_d12_v2"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 9500

    print(f"Loading {ckpt_dir} step {step}...")
    model, tokenizer, meta = build_model(ckpt_dir, step, device, "eval")
    print(f"  bpb: {meta.get('val_bpb', '?')} | CTM: {model.config.use_ctm}")
    print(f"  K={model.config.ctm_iterations} | synapse_depth={model.config.ctm_synapse_depth}")

    # Save original weights for plasticity tests (so we can reset)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Level 1: Cache sanity
    test_cache_sanity(model, tokenizer, device)

    # Level 2: Cache quality comparison
    test_cache_quality(model, tokenizer, device)

    # Level 3: Plasticity (modifies weights!)
    print("\n  [saving model state before plasticity tests]")
    test_plasticity(model, tokenizer, device)

    # Reset weights and try aggressive reinforcement
    print("\n  [resetting model weights]")
    model.load_state_dict(original_state)
    test_plasticity_aggressive(model, tokenizer, device)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
