# nanoctm

fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). we ripped out the MLP layers and replaced them with [Continuous Thought Machines](https://arxiv.org/abs/2505.05522). the model thinks in loops instead of doing one feedforward pass per token.

nobody has tried this on language before. the CTM paper did MNIST and mazes. we wanted to know if it works for the hard modality. it does.

the end goal is a model that learns from conversations and remembers across them. not a stateless tool you prompt and forget — something that accumulates knowledge by being used.

## how it works

normal transformer: attention → MLP → residual. one pass, done.

nanoctm: attention → CTM block → residual. the CTM block runs K iterations of a thinking loop. each iteration re-observes the input through cross-attention (what it pays attention to changes as it thinks more), processes through a depth-32 U-NET synapse, updates a trace of recent states, and accumulates synchronization between random neuron pairs.

after K iterations, each token picks which iteration gave the best answer. not the average, not the last one — each token picks for itself. easy tokens use early ticks, hard tokens use later ones.

the sync signal is real. when neurons agree, certainty is high. when they don't, it's low. this isn't a learned phrase — it's a measurable property of the computation.

`--use-ctm` toggles it. without the flag, it's vanilla nanochat.

## what's running now

d12 CTM on a rented H200 NVL 140GB:

```
12 layers, 768-dim, K=1 (speed chess phase), synapse_depth=32
384 sync neurons, M=16 trace, 32 NLM hidden
~469M params, batch=6 x 5 grad accum, ~3.0s/step, 20k tok/sec
```

step ~2000, loss 3.70, validation bpb ~1.07. model produces sentence fragments
and knows basic facts but still collapses into repetition. learning language
fundamentals before we add thinking iterations.

training history: K=3 (steps 0-1500) → K=5 experiment (150 steps, reverted) → K=1 (current).
see [TRAINING_LOG.md](TRAINING_LOG.md) for the full journey.

## quick start

```bash
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=6 \
    --total-batch-size=61440 \
    --use-ctm \
    --ctm-iterations=1 \
    --ctm-synapse-depth=32 \
    --window-pattern=L \
    --model-tag=ctm_d12_v2 \
    --save-every=500 \
    --sleep-every=10
```

`--window-pattern=L` if you don't have Flash Attention 3. `NANOCHAT_NO_OPTIM_COMPILE=1` if optimizer compile hits recompilation limits. torch.compile works for the full model at K=1.

## K (thinking iterations)

K controls how many times the model thinks about each token. train with K=3, deploy with K=30 — it costs zero extra VRAM at inference because each tick reuses the same buffers. the memory cost is training-only (backprop needs the full computation graph).

the only K-specific parameter is `tick_embed` — a small learned per-tick vector. everything else (synapses, NLMs, sync pairs) is K-agnostic. to scale K after training, expand tick_embed and keep going.

more K can cause overthinking. the model gets it right at tick 3 but ticks 4+ start doubting. per-token selection handles this — doubt-ticks don't get picked. but it means you want to start small and scale up, not train at K=100 from the start.

## training phases

follows brain development. babies learn to talk before they form memories. you need to know what things are before remembering specific instances matters.

**phase 1: language** (now) — from-scratch CTM training. each token gets fresh state, no memory across tokens. the baby learns to talk.

**phase 2: warm-start** (optional) — transplant attention weights from an MLP checkpoint, freeze them, train only CTM blocks. we skipped this — from-scratch works fine and attention co-adapts with CTM better.

**phase 3: memory continuity** — chunked BPTT. split sequences into chunks, process sequentially with CTM state carried between chunks. gradients flow back through chunks — the model learns what to remember for later. without this, feeding carried state at inference is out-of-distribution. it's like handing someone memories from a life they never lived.

**phase 4: constitution SFT** — fine-tune on conversations that embody the [constitution](CONSTITUTION.md). dataset: our Claude Code sessions (~468k tokens), cypherpunk blogs (moxie, djb, isis lovecruft, barlow), penumbra protocol work, local blogs. constitution as system prompt. small dataset, big behavioral shift.

**phase 5: online learning** — gradient step on synapse params after each user message. the conversation is the training data. self-distillation + elastic anchoring prevent forgetting.

**phase 6: episodic memory** — save thinking state snapshots, look up the closest past state when a new conversation starts. resume how you were thinking, not what you said.

**phase 7: metacognitive tokens** — the model reports its own sync/certainty as special tokens. backed by actual internal signals, not learned phrases.

## sleep cycle

every 10 training steps, the model sleeps. three stages:

**REM replay.** replays up to 4 sequences from a buffer of the 16 hardest training examples. softmax-weighted sampling — worst sequences are most likely but not guaranteed. runs dream diagnostics on each: per-layer state deltas, basically an fMRI of the model dreaming.

early finding: the model had the same nightmare every night. one sequence with loss 15.50, so much worse than anything else that nothing displaced it. 13 sleep cycles, 13 replays of the same sequence. consolidation loss halved (2.52 → 1.07), certainty doubled (0.28 → 0.57). it was processing the trauma but never solving it. fixed with diverse replay — re-score entries with fresh forward passes, sample by softmax over losses instead of always picking the worst.

**compaction.** writes sync patterns into permanent synapse weights. hebbian learning — neurons that fire together wire together. novelty-gated: only update where surprise exceeds the median. homeostatic clamping keeps norms stable.

**consolidation.** certainty-weighted self-distillation on 8 sequences (4 hardest + 4 random). confident correct predictions get reinforced. uncertain stuff stays plastic.

`--sleep-every=-1` to disable.

## inference

```bash
# CLI chat
python -m scripts.chat_cli --model-tag=ctm_d12

# web UI
python -m scripts.chat_web --model-tag=ctm_d12

# dream diagnostics
python -m scripts.ctm_dream
```

can't use ollama/llama.cpp/vllm — they don't know how to run CTM loops.

multi-turn sessions:
```python
from nanochat.engine import Session
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", device="cuda", model_tag="ctm_d12")
session = Session(model, tokenizer)
reply1 = session.say("hello")
reply2 = session.say("what did i just say?")
```

online learning (model learns from each message):
```python
session = Session(model, tokenizer, online_lr=1e-5)
session.say("hello")
session.say("my name is tommi")       # learns from this
session.say("what is my name?")       # synapses updated
```

each message is a prediction error signal. CE loss from what the user said vs what the model predicted, KL from pre-update logits to prevent forgetting, L2 toward pretrained weights for stability.

## compute

| config | params | VRAM | step time | tok/sec | hardware |
|--------|--------|------|-----------|---------|----------|
| MLP baseline | 286M | ~10GB | ~0.8s | ~800k | H100 (compiled) |
| CTM K=5 b=2 | 469M | 41GB | 27.0s | 2.4k | H200 NVL |
| CTM K=3 b=2 | 469M | ~110GB | 15.8s | 4.1k | H200 NVL |
| CTM K=1 b=3 | 469M | 65GB | 3.5s | 17.5k | H200 NVL |
| CTM K=1 b=6 | 469M | 120GB | 3.0s | 20.3k | H200 NVL |

CTM is inherently slower than FFN because each layer does 2 attention passes
(self + cross) plus a depth-32 U-NET synapse plus per-neuron MLPs. at K>1 the
entire inner loop repeats sequentially per tick. MFU ~8% vs ~50% for pure FFN.

CTM's sequential computation graph also means it can't scale horizontally across
GPU clusters like FFN can. the scaling path is scale-up (bigger single GPU),
not scale-out (more GPUs). see [TRAINING_LOG.md](TRAINING_LOG.md) for analysis.

## optimizer routing

Muon (polar decomposition) for 2D weight matrices — attention projections, synapse layers, c_proj. AdamW for everything else — 3D NLM weights, 1D params, embeddings. Muon needs 2D matrices, SuperLinear has per-neuron 3D tensors that can't be orthogonalized.

## what we had to invent

the CTM paper and nanochat each work fine alone. combining them broke things neither world had to solve:

- **residual synapses** — paper's synapses are straight-through. at depth 32, gradients vanish. added `state + synapse(obs, state)` — same idea as ResNets, applied to the thinking loop
- **tick embeddings** — without them, all K iterations produce identical outputs because the residual connection dominates at init. learned per-tick signatures force divergence
- **sync seeding** — paper initializes sync accumulators from zeros. for language, tick 0's cross-attention query was `norm(linear(zeros))` — garbage. seed from start_state products instead
- **optimizer routing** — Muon for 2D, AdamW for 3D/1D, zero recompilation
- **dopamine-gated sync** — prediction error scales both alpha and beta accumulators so sync magnitude stays consistent

## what doesn't work (yet)

- **torch.compile at K>1** — graph breaks from Python loops in the tick iteration. K=1 compiles fine. partial compile (attention only) gives ~27% speedup at K>1.
- **multi-GPU training** — DDP adds gradient buffers on top of already heavy activation memory. tried 4×H100, batch=2 OOMed. communication overhead makes it slower than single GPU.
- **persistent CTM state at inference** — trained with fresh state, so carried state is OOD. needs phase 3.
- **online learning** — untested beyond mechanics. might drift.
- **episodic memory** — code works, needs BPTT model.
- **sleep helping** — consolidation loss converges but we don't know if dreaming actually beats just training longer.

## no RLHF

no preference optimization. no reward model. pretraining teaches language — the substrate. values come from a [constitution](CONSTITUTION.md) applied as system prompt, then SFT, then online learning where the model writes experience into its own weights. the model's values come from how it's used, not from how a committee decided it should behave.

## files

```
nanochat/
    gpt.py          # CTMBlock, SynapseUNET, SuperLinear, NLMs, dream/consolidate/compact
    engine.py       # CTMCache, Session, EpisodicMemory
    optim.py        # optimizer routing
    teacher.py      # LocalTeacher, OllamaTeacher for distillation
scripts/
    base_train.py   # training loop with sleep cycle
    ctm_dream.py    # standalone dream diagnostics
data/sft/
    conversations/  # parsed Claude Code sessions
    blogs/          # moxie, djb, isis, barlow, penumbra, etc.
    github/         # penumbra issues, PRs, protocol specs
```

## upstream

fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). see upstream for tokenizer, dataloader, evaluation, chat UI docs.

## license

MIT
