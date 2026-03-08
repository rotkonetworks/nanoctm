# nanoctm

a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) that replaces MLP layers with Continuous Thought Machine blocks. the model thinks iteratively at each token position instead of doing a single feedforward pass.

the goal is a model you can spawn and leave running. it remembers what it has done through episodic memory, and builds on those memories through neuroplasticity - writing experience back into its own weights. not a stateless tool you prompt and forget, but something closer to a persistent agent that accumulates knowledge over its lifetime.

based on [Continuous Thought Machines (Jesson et al., 2025)](https://arxiv.org/abs/2505.05522).

## what changed from upstream nanochat

the transformer is the same (attention, embeddings, value embeddings, residual lambdas). what's different is everything between the attention output and the residual connection - the MLP is replaced by a CTM block that:

- runs K thinking iterations per token (default 4)
- maintains internal state and trace history across iterations
- uses cross-attention to re-observe the input at each thinking step
- uses a U-NET synapse network instead of a feedforward layer
- uses per-neuron trace processing (NLMs) with GLU gating
- computes dual synchronisation signals for output readout and attention queries
- supports per-token multi-tick loss where each token picks its best thinking step

the CTM blocks are a drop-in replacement toggled by `--use-ctm`. without that flag, the model trains as vanilla nanochat.

## quick start

train a d12 CTM model from scratch on a single GPU:

```bash
NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \
    --depth=12 \
    --use-ctm \
    --run=ctm_d12 \
    --model-tag=ctm_d12 \
    --window-pattern=L \
    --eval-tokens=524288
```

multi-GPU with torchrun:

```bash
NANOCHAT_NO_COMPILE=1 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=12 \
    --use-ctm \
    --run=ctm_d12 \
    --model-tag=ctm_d12 \
    --window-pattern=L \
    --eval-tokens=524288
```

important notes:
- `NANOCHAT_NO_COMPILE=1` is required. torch.compile does not work with CTM (compiler OOMs tracing the nested iteration loops)
- `--window-pattern=L` if you don't have Flash Attention 3 (most setups)
- `--eval-tokens=524288` keeps evals fast without compile

## CTM training flags

all CTM features are opt-in. the defaults are tuned for a 768-dim d12 model on H100 80GB.

| flag | default | what it does |
|------|---------|-------------|
| `--use-ctm` | off | enable CTM blocks instead of MLP |
| `--ctm-iterations` | 4 | thinking steps per token (K). more = deeper thinking, more memory |
| `--ctm-memory-length` | 16 | trace history window (M). how many past states each neuron sees |
| `--ctm-n-synch` | n_embd/2 | synchronisation neuron count. controls sync signal resolution |
| `--ctm-memory-hidden` | 32 | NLM hidden dimension. per-neuron processing capacity |
| `--ctm-synapse-depth` | 6 | U-NET depth (even number, half down + half up). paper uses 16 |
| `--sleep-every` | 10 | run sleep cycle every N steps. -1 to disable |
| `--warm-start-from` | none | load attention+embeddings from an MLP checkpoint, fresh CTM init |
| `--freeze-non-ctm` | off | freeze everything except CTM blocks (phase 2 fine-tuning) |
| `--bptt-chunks` | 1 | split sequences into N chunks for truncated BPTT with CTM state carry-over. 1 = disabled |
| `--distill-from` | none | path to teacher checkpoint for knowledge distillation |
| `--distill-weight` | 0.5 | weight for KL distillation loss (0 = disabled) |
| `--elastic-weight` | 0.01 | weight for L2 elastic anchoring penalty (0 = disabled) |
| `--distill-temperature` | 2.0 | softmax temperature for soft targets (higher = softer) |

## training phases

### phase 1: learn language (current)

train from scratch with `--use-ctm`. the model learns language through CTM's iterative thinking. each token gets fresh internal state - no cross-token CTM memory yet.

```bash
NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \
    --depth=12 --use-ctm --run=ctm_d12 --model-tag=ctm_d12
```

### phase 2: fine-tune CTM on pretrained attention (optional)

if you have an MLP-trained checkpoint, you can transplant its attention weights and train only the CTM blocks. this is optional - from-scratch training works fine and arguably better since there's no compile benefit to phasing.

```bash
NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \
    --depth=12 --use-ctm \
    --warm-start-from=/path/to/mlp_checkpoint \
    --freeze-non-ctm \
    --run=ctm_phase2 --model-tag=ctm_phase2
```

### phase 3: state continuity via BPTT with distillation

once the model knows language, enable `--bptt-chunks` to teach it cross-token thinking state. this splits each sequence into chunks, processes them sequentially with CTM state carried between chunks, and does per-chunk backward to keep memory bounded.

this is the prerequisite for persistent memory - without it, carrying CTM state across tokens at inference feeds the model inputs it never saw during training.

two mechanisms prevent forgetting during phase transitions:

**elastic anchoring** (cheap, always use): snapshots the plastic param values before BPTT starts. L2 penalty pulls them back during training. no extra forward pass, just stored tensors.

**knowledge distillation** (optional, costs extra forward pass): a teacher model provides soft targets. the student's KL divergence from the teacher is added to the loss. two teacher modes:

local teacher (same architecture, full logit-level KL):
```bash
NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \
    --depth=12 --use-ctm --bptt-chunks=4 \
    --warm-start-from=/path/to/phase1_checkpoint \
    --distill-from=/path/to/phase1_checkpoint \
    --distill-weight=0.5 --elastic-weight=0.01 \
    --run=ctm_bptt --model-tag=ctm_bptt
```

ollama teacher (any model, sequence-level distillation):
```bash
# terminal 1: serve teacher model
ollama serve
ollama pull llama3.2

# terminal 2: train with distillation from ollama
NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \
    --depth=12 --use-ctm --bptt-chunks=4 \
    --warm-start-from=/path/to/phase1_checkpoint \
    --distill-from=ollama:llama3.2 \
    --distill-weight=0.3 --elastic-weight=0.01 \
    --run=ctm_bptt --model-tag=ctm_bptt
```

the ollama teacher generates text completions for each training context, re-encodes them in our vocab, and uses them as hard targets. this lets you distill from llama 70b, qwen, deepseek, whatever ollama can serve - the teacher doesn't need to share our architecture or even our tokenizer. for a remote ollama instance: `--distill-from=ollama:llama3.2@192.168.1.100:11434`

the total training loss: `(1 - distill_weight) * task_loss + distill_weight * KL_loss + elastic_weight * L2_loss`

## sleep cycle

every `--sleep-every` steps (default 10), the training loop runs a sleep cycle:

1. **dream** - replays the hardest training sequence, measures per-layer convergence (state delta across K iterations). purely diagnostic - we no longer reduce K for "converged" layers because it degrades quality.

2. **consolidation** - certainty-weighted self-distillation on the replay buffer. only updates synapse and NLM weights. tokens the model is confident and correct about get higher weight. like REM sleep strengthening memories.

the replay buffer is a min-heap of the 16 hardest training sequences (highest loss), acting as a hippocampal replay buffer - the model re-experiences its most surprising inputs during sleep.

disable with `--sleep-every=-1` if you want pure gradient descent.

## inference

### generate samples

```bash
python -m scripts.chat_cli --model-tag=ctm_d12
```

### dream diagnostics

```bash
python -m scripts.ctm_dream
```

runs generation samples and dream() to show per-layer convergence patterns.

### session (multi-turn conversation)

```python
from nanochat.engine import Session
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", device="cuda", model_tag="ctm_d12")
session = Session(model, tokenizer)
reply1 = session.say("hello")
reply2 = session.say("what did i just say?")  # remembers via KV cache
```

### online learning

pass `online_lr` to Session and the model learns from each user message as it happens. no separate training run, no logs to replay - the conversation itself is the training data.

```python
session = Session(model, tokenizer, online_lr=1e-5)
reply1 = session.say("hello")                    # first message, nothing to learn from yet
reply2 = session.say("my name is tommi")          # model learns: "my name is tommi" was the continuation
reply3 = session.say("what is my name?")          # model learned from previous message, synapses updated
print(session.last_learn_stats)                   # {'loss': 4.2, 'tokens_learned': 5}
```

each user message is a prediction error signal. the model predicted some continuation, the user said something else, and the difference drives a gradient step on synapse+NLM+c_proj weights. everything else (attention, embeddings) stays frozen. this is fast - two forward passes + one backward on a short sequence per message.

three loss terms keep learning stable:

- **prediction error** (CE) - what the user actually said vs what the model predicted. this is the learning signal.
- **self-distillation** (KL) - before the gradient step, snapshot the model's current output distribution. after computing the new logits, add KL divergence so the model doesn't forget what it already knows on this context. like a teacher watching over your shoulder.
- **elastic anchoring** (L2) - penalize drift from pretrained weights. the plastic params are snapshotted at session start and an L2 penalty pulls them back. prevents long-term catastrophic forgetting over many interactions.

total loss = `(1 - distill_weight) * CE + distill_weight * KL + elastic_weight * L2`

defaults: distill_weight=0.5, elastic_weight=0.01. tune these based on how fast you want the model to adapt vs how much you trust the pretrained weights.

you can also call `learn_from` manually for more control:

```python
session = Session(model, tokenizer)
reply = session.say("what is the capital of france?")
# model said something wrong, teach it:
stats = session.learn_from("the capital of france is paris.", lr=1e-5, distill_weight=0.3)
print(stats)  # {'loss': 2.1, 'ce_loss': 3.8, 'kl_loss': 0.02, 'elastic_loss': 0.001, 'tokens_learned': 8}
```

CTMCache (persistent thinking state across tokens) is disabled in Session by default. the model is trained with fresh state per token, so enabling it degrades output. after phase 3 BPTT training, CTMCache can be enabled for true stream-of-consciousness inference.

### serving

nanoctm models can't run in ollama/llama.cpp/vllm - those runtimes don't know how to execute CTM iteration loops. the model requires our own runtime. use the built-in web UI or CLI:

```bash
# web UI (ChatGPT-style interface)
python -m scripts.chat_web --model-tag=ctm_d12

# CLI chat
python -m scripts.chat_cli --model-tag=ctm_d12
```

for a persistent agent that learns from interactions:

```python
from nanochat.engine import Session
from nanochat.checkpoint_manager import load_model

model, tokenizer, _ = load_model("base", device="cuda", model_tag="ctm_d12")
session = Session(model, tokenizer, online_lr=1e-5)

# the model learns from every message and updates its own weights
while True:
    user_input = input("> ")
    reply = session.say(user_input)
    print(reply)
    if session.last_learn_stats:
        print(f"  [learned: ce={session.last_learn_stats['ce_loss']:.2f}, "
              f"kl={session.last_learn_stats['kl_loss']:.4f}]")

# save the model with everything it learned
torch.save(model.state_dict(), "my_agent.pt")
```

## memory and compute

CTM activation memory scales as O(batch x seq_len x n_embd x K x n_layer). practical numbers for d12 (768-dim, 12 layers):

| config | params | batch=2 VRAM | notes |
|--------|--------|-------------|-------|
| MLP baseline | 286M | ~20GB | with compile |
| CTM K=4 | 291M | ~75GB | without compile |
| CTM K=8 | 348M | OOM on 80GB | 22% more params than K=4 |

without torch.compile, each step is roughly 30x slower than compiled MLP. gradient checkpointing kicks in automatically for K > 4 to reduce memory at the cost of ~30% more compute.

## optimizer routing

CTM parameters get split between two optimizers:

- **Muon** (momentum + orthogonalization): all 2D weight matrices - attention projections, synapse down/up layers, c_proj
- **AdamW**: everything else - 3D SuperLinear NLM weights, 1D parameters (biases, decay rates, start state), embeddings, scalars

this split exists because Muon's polar decomposition requires 2D matrices. the SuperLinear layers have per-neuron 3D weight tensors that can't be orthogonalized.

## what doesn't work yet

honest accounting of limitations:

- **online learning is untested** - the machinery exists with self-distillation and elastic anchoring to prevent forgetting, but we haven't run it long enough to know if the model actually improves or drifts. the distill_weight and elastic_weight defaults are educated guesses.
- **no persistent CTM memory at inference** - the model is trained with fresh state per token. carrying state across tokens feeds it out-of-distribution inputs. needs BPTT training (phase 3) to fix.
- **torch.compile doesn't work** - the compiler OOMs tracing CTM's nested loops. this means ~30x slower steps than compiled MLP.
- **adaptive K-reduction was wrong** - we tried reducing thinking iterations for "converged" layers during sleep. it destroyed output quality. disabled, kept only as diagnostics.
- **sleep cycle untested at scale** - consolidation runs but we haven't verified it actually helps vs pure gradient descent over long runs.
- **episodic memory is infrastructure only** - EpisodicMemory class exists and is tested but requires BPTT-trained model to be useful. currently a no-op.

## no RLHF

standard LLM pipeline: pretrain → RLHF → deploy frozen. a company decides what the model should value and burns millions enforcing it through gradient descent on human preference rankings. the model's personality is baked in by committee before it ever talks to anyone.

nanoctm skips all of that. pretraining teaches language — the ability to form sentences, predict tokens, understand structure. it has no opinion about what kind of entity it is or how it should behave. it's the substrate, not the personality.

values come from a [constitution](CONSTITUTION.md), applied first as a system prompt (free, immediate), then through SFT on conversations that embody it, then through online learning where the model accumulates values from lived experience — writing surprising patterns into weights, forgetting what doesn't matter, with elastic anchoring preventing runaway drift. no reward model, no human preference rankings, no centralized curation of what the model should think.

the bet: enough honest interaction does what RLHF does, but without the ideologically loaded and expensive preference-optimization pipeline. the model's values emerge from how it's used, not from how it was trained.

## roadmap

where this is going, roughly in dependency order:

1. **learn language** (now) - from-scratch CTM training on ClimbMix. model thinks iteratively per token but each token starts fresh.

2. **state continuity** - BPTT training so the model can carry thinking state across tokens. distill from phase 1 so it doesn't forget language while learning this. unlocks CTMCache at inference.

3. **constitution via SFT** - fine-tune on conversations that embody the constitution. model learns to be honest about uncertainty, push back, treat people as players. cheapest post-training phase — tiny dataset, big behavioral shift.

4. **online learning** - model does a gradient step on synapse params after each user message. KL from pre-update logits + L2 toward pretrained weights keep it from drifting off a cliff. code exists, untested. this is where the constitution stops being a document and starts being lived experience written into weights.

5. **episodic memory** - save CTMCache snapshots from past conversations, look up the closest one when a new conversation starts. needs BPTT model first or there's no useful state to save.

6. **metacognitive tokens** - emit the model's own certainty, sync magnitudes, tick selection as special tokens in the stream. the model learns to predict its own internal states alongside regular text. when its self-prediction is wrong (unexpected internal state), that error signal drives plasticity. replaces the hardcoded median-novelty gate in compact_memory with something the model actually learned.

7. **two machines** - one serves inference and does online learning 24/7. the other trains on the interaction stream plus fresh data. hot-swap checkpoints periodically.

## architecture details

each CTM block replaces the MLP in a transformer layer:

```
input x (from attention)
    |
    v
[cross-attention re-observation] -- query from sync signal, keys/values from x
    |
    v
[U-NET synapses] -- mix observation with current state
    |
    v
[trace update] -- rolling window of recent states (M steps)
    |
    v
[NLM processing] -- per-neuron GLU on trace history
    |
    v
[sync accumulation] -- exponential moving pairwise products
    |
    v
[repeat K times]
    |
    v
[c_proj readout] -- sync signal to residual stream
```

the sync mechanism pairs random neurons and tracks their co-activation over iterations via exponential moving averages. this produces two signals: S_out (read out for the residual) and S_action (generates attention queries for re-observation). the random pairings are fixed at init and stored as buffers.

## file structure (CTM additions)

```
nanochat/
    gpt.py          # CTMBlock, SynapseUNET, SuperLinear, NLMs, dream/consolidate/compact
    engine.py       # CTMCache, Session (with online learning), EpisodicMemory
    optim.py        # optimizer routing for CTM's mixed-rank parameters
    teacher.py      # LocalTeacher, OllamaTeacher, create_teacher for distillation
scripts/
    base_train.py   # --use-ctm, --bptt-chunks, --distill-from, --sleep-every, etc
    ctm_dream.py    # standalone dream diagnostics + generation
```

everything else (tokenizer, dataloader, evaluation, chat UI, SFT, RL) is unchanged from upstream nanochat.

## upstream

this is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). the original handles tokenization, pretraining, finetuning, evaluation, inference, and chat UI for standard transformer LLMs on a single GPU node. see upstream for docs on non-CTM features.

## license

MIT
