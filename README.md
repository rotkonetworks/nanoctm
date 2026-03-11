# nanoctm

fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). we ripped out the MLP layer in the final transformer block and replaced it with a [Continuous Thought Machine](https://arxiv.org/abs/2505.05522). the model thinks in loops instead of doing one feedforward pass per token.

nobody has tried this on language before. the CTM paper did image classification and mazes. we're the first to attempt CTM for autoregressive text generation.

the end goal is a model that learns from conversations and remembers across them. not a stateless tool you prompt and forget — something that accumulates knowledge by being used.

## architecture

normal transformer: attention → MLP → residual. one pass, done.

nanoctm: layers 0-10 are normal (attention → MLP). layer 11 is attention → CTM block → residual. the CTM block runs K iterations of a thinking loop. each iteration re-observes the input through cross-attention, processes through a depth-32 U-NET synapse, updates a trace of recent states, and accumulates synchronization between random neuron pairs.

after K iterations, each token picks which iteration gave the best answer. not the average, not the last one — each token picks for itself. easy tokens use early ticks, hard tokens use later ones.

**why single CTM?** the paper uses one CTM with 50-100 ticks on top of a backbone network. we tried 12 CTMs (one per layer) and discovered why the paper didn't do that — 12 independent thinkers can't build coherent thought. 11 untrained CTMs corrupt the signal before the last layer sees it. single CTM on the last layer: 27x faster, coherent generation, clean plasticity signal. `--ctm-layers=last` controls this.

`--use-ctm` toggles CTM. without the flag, it's vanilla nanochat.

## what's running now

d12 single-CTM on a rented H200 NVL 140GB:

```
12 layers (11 FFN + 1 CTM), 768-dim, K=1 with auto-ramp
synapse_depth=32, 384 sync neurons, M=16 trace, 32 NLM hidden
~521M params, batch=3 x 10 grad accum, ~850ms/step, 72k tok/sec
cache-aware training at 30%, scheduled sampling at 10%
```

warm-started from FFN 10k checkpoint (bpb 0.908). layers 0-10 keep trained MLP weights, layer 11 starts fresh CTM. val bpb ~1.024 at step 2200.

generation with CTMCache works — first time ever:
```
"The meaning of life is well known in many respects. It was once one
of many reasons for the exploitation of life in India, China..."
```

auto K-ramp: plateau detection monitors val bpb. when loss stalls, K bumps automatically. each K level adds thinking depth to one coherent CTM.

see [TRAINING_LOG.md](TRAINING_LOG.md) for the full journey.

## quick start

```bash
# single-CTM training (recommended)
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=3 \
    --total-batch-size=61440 \
    --use-ctm \
    --ctm-iterations=1 \
    --ctm-synapse-depth=32 \
    --ctm-layers=last \
    --window-pattern=L \
    --cache-aware-ratio=0.30 \
    --scheduled-sampling=0.10 \
    --k-ramp=plateau --k-max=50 \
    --model-tag=ctm_d12_single \
    --save-every=500 \
    --sleep-every=10

# warm-start from FFN checkpoint
python -m scripts.base_train \
    --use-ctm --ctm-layers=last \
    --warm-start-from=/path/to/ffn_checkpoint \
    ...
```

`--window-pattern=L` if you don't have Flash Attention 3. `NANOCHAT_NO_OPTIM_COMPILE=1` if optimizer compile hits recompilation limits. torch.compile works for the full model at K=1.

## K (thinking iterations)

K controls how many times the model thinks about each token. start with K=1, ramp up when loss plateaus — `--k-ramp=plateau` does this automatically.

the only K-specific parameter is `tick_embed` — a small learned per-tick vector. everything else (synapses, NLMs, sync pairs) is K-agnostic. to scale K after training, expand tick_embed and keep going.

more K can cause overthinking. the model gets it right at tick 3 but ticks 4+ start doubting. per-token selection handles this — doubt-ticks don't get picked. but it means you want to start small and scale up, not train at K=100 from the start.

## plasticity (the point of all this)

the CTM accumulates sync statistics during inference — which neurons fire together. `compact_memory()` writes these patterns into permanent synapse weights. hebbian learning: neurons that fire together wire together.

```python
engine = Engine(model, tokenizer)
fact_tokens = tokenizer.encode("My name is Tommi. I live in Helsinki.")
results, masks, stats = engine.generate_and_compact(
    fact_tokens, max_tokens=20, plasticity_lr=1e-4
)
# synapse weights permanently updated — model remembers
```

current status: mechanism works (weights change), but at K=1 the sync signal is too weak to affect output. needs K >> 1 for the CTM to build meaningful patterns. this is expected — the paper uses 50-100 ticks. we're ramping K incrementally.

## training phases

follows brain development. babies learn to talk before they form memories.

**phase 1: language** (current) — FFN baseline (10k steps), then warm-start single CTM with auto K-ramp. cache-aware training from step 0 so CTMCache works at inference.

**phase 2: memory continuity** — chunked BPTT. split sequences into chunks, process sequentially with CTM state carried between chunks. gradients flow back through chunks — the model learns what to remember for later.

**phase 3: constitution SFT** — fine-tune on conversations that embody the [constitution](CONSTITUTION.md). dataset: our Claude Code sessions, cypherpunk blogs, local blogs. constitution as system prompt. small dataset, big behavioral shift.

**phase 4: online learning** — gradient step on synapse params after each user message. the conversation is the training data. self-distillation + elastic anchoring prevent forgetting.

**phase 5: episodic memory** — save thinking state snapshots, look up closest past state when a new conversation starts. resume how you were thinking, not what you said.

**phase 6: metacognitive tokens** — the model reports its own sync/certainty as special tokens. backed by actual internal signals, not learned phrases.

## sleep cycle

every 10 training steps, the model sleeps. three stages:

**REM replay.** replays up to 4 sequences from a buffer of the 16 hardest training examples. softmax-weighted sampling — worst sequences are most likely but not guaranteed. runs dream diagnostics on each.

**compaction.** writes sync patterns into permanent synapse weights. novelty-gated: only update where surprise exceeds the median. homeostatic clamping keeps norms stable.

**consolidation.** certainty-weighted self-distillation on 8 sequences (4 hardest + 4 random). confident correct predictions get reinforced. uncertain stuff stays plastic.

`--sleep-every=-1` to disable.

## compute

| config | params | VRAM | step time | tok/sec | hardware |
|--------|--------|------|-----------|---------|----------|
| FFN baseline (compiled) | 286M | ~10GB | ~0.4s | ~150k | H200 NVL |
| 12 CTMs K=3 b=2 | 892M | ~110GB | 21.0s | 2.9k | H200 NVL |
| **1 CTM K=1 b=3** | **521M** | **~30GB** | **0.85s** | **72k** | **H200 NVL** |

single CTM is **27x faster** than 12 CTMs. FFN layers 0-10 run at full speed, only layer 11 pays the CTM cost.

## inference

```bash
# CLI chat
python -m scripts.chat_cli --model-tag=ctm_d12_single

# web UI
python -m scripts.chat_web --model-tag=ctm_d12_single
```

can't use ollama/llama.cpp/vllm — they don't know how to run CTM loops.

## what we had to invent

the CTM paper and nanochat each work fine alone. combining them broke things neither world had to solve:

- **single-CTM architecture** — paper uses one CTM, not per-layer. 12 CTMs fragment coherence. `--ctm-layers=last` puts CTM only on the final layer
- **cache-aware training** — CTMCache accumulates state across tokens. train with 30% of batches seeing accumulated cache so the model isn't surprised by it at inference
- **residual synapses** — paper's synapses are straight-through. at depth 32, gradients vanish. added `state + synapse(obs, state)`
- **tick embeddings** — without them, all K iterations produce identical outputs. learned per-tick signatures force divergence
- **sync seeding** — paper initializes sync from zeros. for language, that makes tick 0's cross-attention garbage. seed from start_state products instead
- **auto K-ramp** — plateau detection bumps K automatically. resizes tick_embed, resets optimizer, recompiles
- **scheduled sampling** — feeds model its own predictions during training to handle autoregressive generation errors
- **optimizer routing** — Muon for 2D matrices, AdamW for 3D/1D params, zero recompilation

## what doesn't work (yet)

- **torch.compile at K>1** — graph breaks from Python loops in the tick iteration. K=1 compiles fine
- **multi-GPU training** — DDP adds gradient buffers on top of already heavy activation memory
- **plasticity at K=1** — sync signal too weak. needs higher K for meaningful compact_memory
- **conversational ability** — raw pretrained model, no SFT yet. generates coherent text but can't hold a conversation

## files

```
nanochat/
    gpt.py              # CTMBlock, SynapseUNET, SuperLinear, NLMs, compact_memory, dream
    engine.py           # CTMCache, Engine, Session, EpisodicMemory
    optim.py            # Muon/AdamW routing
    checkpoint_manager.py  # save/load with mixed MLP/CTM support
scripts/
    base_train.py       # training loop with sleep cycle, K-ramp, cache-aware training
    ctm_dream.py        # standalone dream diagnostics
knowledge/
    summary_ctm_paper.md  # CTM paper analysis and implications for nanochat
```

## upstream

fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). see upstream for tokenizer, dataloader, evaluation, chat UI docs.

## license

MIT
