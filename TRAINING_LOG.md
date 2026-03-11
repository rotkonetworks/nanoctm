# training log

a record of our model's journey from noise to something that thinks.

## contents

- [step 0 — launch](#step-0--launch-2026-03-07)
- [step 500 — first checkpoint](#step-500--first-checkpoint-2026-03-08)
- [step 1000 — it knows words](#step-1000--it-knows-words-2026-03-08)
- [migration — 4x H100 attempt](#migration--4x-h100-attempt-2026-03-09)
- [migration — H200 NVL 140GB](#migration--h200-nvl-140gb-2026-03-09)
- [step 1150+ — current](#step-1150--current-2026-03-09)
- [step 1500 — K=3 → K=5 switch](#step-1500--k3--k5-switch-2026-03-09)
- [stats at a glance](#stats-at-a-glance)
- [step 1500 — K=5 → K=1 switch](#step-1500--k5--k1-switch-2026-03-09)
- [step 1500 — K=1 batch scaling](#step-1500--k1-batch-scaling-2026-03-09)
- [FFN baseline run](#ffn-baseline-run-2026-03-09)
- [FFN baseline to 20k](#ffn-baseline-to-20k-2026-03-09)
- [project isis — frankenstein warm-start](#project-isis--frankenstein-warm-start-2026-03-10)
- [the plan: proof of concept roadmap](#the-plan-proof-of-concept-roadmap)
- [from-scratch CTM: the better path](#from-scratch-ctm-the-better-path-2026-03-10)
- [upstream optimizations cherry-picked](#upstream-optimizations-cherry-picked-step-55007000)
- [step 7000 — CORE benchmark + generation](#step-7000--core-benchmark--generation-analysis-2026-03-10)
- [step 7000→8500 — K=2 training](#step-70008500--k2-training-2026-03-10)
- [step 8000→ — K=4 training](#step-8000--k4-training-2026-03-10)
- [the generation problem — diagnosis and fix](#the-generation-problem--diagnosis-and-fix-2026-03-10)
- [step 8000 — K=2 + scheduled sampling](#step-8000--k2--scheduled-sampling-2026-03-10)
- [CRITICAL: CTMCache breaks generation](#critical-finding-ctmcache-breaks-generation-2026-03-10)
- [clean FFN d12 run](#clean-ffn-d12-run-2026-03-10)
- [state of affairs — honest assessment](#state-of-affairs--an-honest-assessment-2026-03-10)
- [step 9000→9500 — cache-aware, flat 0.3](#step-90009500--cache-aware-training-flat-03-2026-03-10)
- [step 9000 (take 2) — adaptive ramp](#step-9000-take-2--adaptive-cache-aware-ramp-2026-03-10)
- [ramp tuning and restart saga](#ramp-tuning-and-restart-saga-2026-03-11)
- [step 10000 — plasticity test](#step-10000--cache-aware-ramp-complete-plasticity-test-2026-03-11)
- [step 10200 — tick analysis](#step-10200--ramp-at-030-tick-analysis-2026-03-11)
- [step 11000→11500 — K=3 bump, generation collapse](#step-1100011500--k3-bump-generation-collapse-2026-03-11)
- [the breakthrough: single CTM architecture](#the-breakthrough-single-ctm-architecture-2026-03-11)
- [single CTM: step 1500-2200, generation works, plasticity weak](#single-ctm-step-1500-2200-generation-works-plasticity-weak-2026-03-11)

## step 0 — launch (2026-03-07)

set sail. d12 CTM, 768-dim, 12 layers, 469M params. K=3 thinking iterations,
synapse_depth=32, memory_length=16, 384 synch neurons.

training on 1x H100 80GB (Vast.ai), batch=2, 16 grad accum steps, no torch.compile
(compiler OOMs on CTM's nested loops). ~21.5s/step, 3k tok/sec.

data: climbmix-400b-shuffle, 170 shards (~15GB). learning rate ramping up.
token ratio: 0.7 tokens/param. chinchilla scaling doesn't apply to CTMs —
K iterations per token means fundamentally different compute economics.
we need to discover our own dogu optimal scaling.

## step 500 — first checkpoint (2026-03-08)

loss: 4.25. ticks loss=[5.8, 4.5, 4.3] cert=[0.42, 0.60, 0.63].
model knows tokens exist. not much else.

deployed diverse sleep cycle here. before this, sleep was stuck replaying
one nightmare over and over. fixed with softmax-weighted sampling over replay
buffer, re-scoring with fresh forward passes, 4 dreams per cycle.

backed up to 160.22.181.7.

## step 1000 — it knows words (2026-03-08)

loss: 3.57. validation bpb: 1.14.

generation test (brute-force, no cache):
- "The meaning of life is" → "life life life life..."
- "The president of the United States" → "is a a a game game game..."
- "Python is a programming language that" → "can help support support support..."

it picks contextually relevant english words but gets stuck in repetition.
knows basic grammar ("is a", "that can help"). no sentence structure yet.

tick diagnostics: loss=[4.77, 3.64, 3.51] cert=[0.51, 0.69, 0.71]
tick 2 selected 50-51% of the time. all 12 layers actively thinking.

sleep: replay buffer healthy, 16 entries, loss range [2.89, 4.39].
consolidation loss ~0.75, certainty ~0.65.

backed up to 160.22.181.7. original H100 instance died shortly after.

## migration — 4x H100 attempt (2026-03-09)

tried 4x H100 80GB for multi-GPU training. CTM's iterative computation
doesn't shard well across GPUs — DDP adds gradient buffers on top of
already heavy activation memory. batch=2 OOMed on 80GB per card.
batch=1 was slower than single GPU due to communication overhead.

fixed some bugs along the way: optimizer reduce_scatter needs tensor dims
divisible by world_size, CTM has odd-shaped params. added all_reduce fallback.

moved on to bigger single GPU.

## migration — H200 NVL 140GB (2026-03-09)

moved to 1x H200 NVL, 140GB VRAM, Xeon 6747P 192 cores, 2TB RAM. $1.95/hr.

torch.compile attempted on the full model. compiler OOMed during tracing on the
old instance (compile uses system RAM, not VRAM). this machine has 2TB system
RAM so tracing completes, but the CTMBlock still isn't fully compiled — Python
loops and branches cause graph breaks. attention modules compile, CTM blocks
stay interpreted.

gotchas along the way:
- 32GB overlay disk filled up from training data + pip cache. moved data to
  /dev/shm (125GB), cleared pip cache. compile cache needs executable storage
  (not /dev/shm which is noexec).
- muon_step_fused hits recompile limit with fullgraph=True. added
  NANOCHAT_NO_OPTIM_COMPILE env var to skip just optimizer compile.

results at step ~1150:
- 15.8s/step, 4.1k tok/sec (was 21.5s on old H100 — 27% faster)
- 110GB VRAM used, 30GB headroom
- loss ~3.75 (slightly higher than old run at same step — fresh optimizer)

tick selection locked at: tick0=28%, tick1=23%, tick2=49%. not shifting.
this suggests K=3 is a ceiling — model can't differentiate further.

plan: save checkpoint 1500, bump K from 3 to 5. more ticks = more room
for the model to develop tick specialization (rough → refined answers).

## step 1150+ — current (2026-03-09)

training steady. sleep cycles healthy:
- 4 diverse dreams per cycle, losses ranging 2.5-3.8
- replay buffer: 16 entries, max replays: 6
- consolidation loss: ~0.73, certainty: ~0.64
- 0/12 layers converged — all still actively iterating

waiting for step 1500 checkpoint to switch to K=5.

## step 1500 — K=3 → K=5 switch (2026-03-09)

loss: 3.56. validation bpb: 1.075.

switched from K=3 to K=5 thinking iterations. tick_embed expanded from
(3, 768) to (5, 768) — existing 3 ticks preserved, new 2 initialized
with small random noise. optimizer state discarded (momentum buffers
shaped for K=3 incompatible with K=5).

first step diagnostics with K=5:
  loss=[5.19, 4.12, 3.95, 3.92, 4.17] cert=[0.47, 0.62, 0.65, 0.61, 0.61]
  selected=[30%, 13%, 45%, 3%, 8%]

tick 2 still dominates (45%). ticks 3-4 are untrained — tick 4 loss actually
rises above tick 3 (4.17 vs 3.92). they'll need training to find their role.

step time: 15.8s → 27s (1.7x for 1.67x more ticks — proportional).
VRAM: 41GB / 140GB — plenty of headroom.

generation test at step 1500 (K=3 checkpoint, before switch):
- "Python is a programming language that" → "is not not only language but not language. In this article..."
- "Water boils at 100 degrees" → "C C C C..."
- "The quick brown fox jumped over" → "over over over..."

improvement from step 1000: actual sentence fragments now ("is not only language
but not language"), correct facts ("100 degrees C"). still collapses into
repetition after 5-10 tokens. loss needs to drop more before coherent output.

### K=5 tick evolution (150 steps of data)

```
step  tick losses                          tick selection
1500  [5.19, 4.12, 3.95, 3.92, 4.17]     [30%, 13%, 45%,  3%,  8%]  ← first K=5 step
1510  [4.48, 3.75, 3.75, 3.88, 3.74]     [24%, 11%, 36%, 17%, 12%]
1520  [4.72, 4.16, 3.81, 4.02, 3.72]     [17%, 13%, 33%, 20%, 17%]
1530  [4.51, 3.90, 3.90, 3.96, 3.85]     [27%,  9%, 32%, 17%, 15%]
1540  [4.44, 3.66, 3.56, 3.70, 3.52]     [26%, 11%, 33%, 17%, 13%]
1640  [5.29, 4.31, 4.18, 4.21, 4.25]     [29%, 10%, 30%, 19%, 13%]
1650  [4.64, 3.83, 3.74, 3.79, 3.74]     [25%, 11%, 32%, 17%, 16%]
```

new ticks learned fast: tick 4 went from worst (loss 4.17, 8% selected) to
competitive (3.74, 16%) in ~150 steps. tick 3 went from 3% → 17% selection.
tick 2 dominance dropped from 45% → 32% as load spread across all 5 ticks.

but the overall loss barely moved. K=3 at step 1500: loss 3.56, bpb 1.075.
K=5 at step 1650: loss 3.49, bpb ~1.066. that's 0.009 bpb improvement in
150 steps — same rate as K=3 was achieving (0.007/100 steps). the extra
ticks didn't accelerate learning, they just spread the same work thinner.

step time cost: 15.8s (K=3) → 27s (K=5) = 1.7x slower per step.
for the same wall-clock time, K=3 would have done ~250 steps vs K=5's 150.
K=5 is strictly worse at this stage of training.

### verdict: K=5 premature, switching to K=1

the model is still learning basic language — token co-occurrence, grammar,
simple facts. it doesn't need deep thinking yet. every extra tick is pure
overhead during this phase. tick 0 (the "gut instinct" tick) was already
getting more confident over K=3 training (cert 0.51 → 0.56), meaning the
model's first guess was improving and later ticks had diminishing returns.

**scaling insight**: base training should start with K=1 — pure FFN-style,
maximum throughput, bruteforce language learning. only add thinking ticks
later once the model has basic language down and needs to reason. K>1 during
early training slows step time proportionally for something the model can't
yet use. we went K=3 from step 0, costing ~1.7x in step time. K=5 made it
2.7x. the right move is K=1 until the model speaks coherently, then ramp up.

switching to K=1 with batch=6 from checkpoint 1500. this is the dogu optimal
strategy: fast base, deep thinking later.

---

## stats at a glance

| step | loss | bpb   | tick losses        | tick certs         | tick selection    | sleep consol | replay range |
|------|------|-------|--------------------|--------------------|-------------------|-------------|--------------|
| 0    | ~11  | —     | —                  | —                  | —                 | —           | —            |
| 500  | 4.25 | —     | [5.8, 4.5, 4.3]   | [0.42, 0.60, 0.63] | —                | —           | —            |
| 1000 | 3.57 | 1.114 | [4.77, 3.64, 3.51] | [0.51, 0.69, 0.71] | [25%, 26%, 49%]  | 0.75        | [2.89, 4.39] |
| 1100 | 3.73 | 1.110 | [5.45, 4.27, 4.13] | [0.49, 0.66, 0.68] | [27%, 25%, 48%]  | 0.71        | [2.81, 3.77] |
| 1200 | 3.65 | 1.098 | [5.13, 4.02, 3.95] | [0.51, 0.67, 0.70] | [27%, 23%, 49%]  | 0.66        | [2.56, 3.42] |
| 1300 | 3.61 | 1.089 | [5.43, 4.28, 4.16] | [0.49, 0.64, 0.67] | [30%, 23%, 47%]  | 0.78        | [2.67, 4.34] |
| 1400 | 3.55 | 1.080 | [5.10, 4.09, 3.95] | [0.51, 0.66, 0.68] | [27%, 24%, 49%]  | 0.76        | [2.99, 4.30] |
| 1500 | 3.56 | 1.075 | [4.64, 3.80, 3.71] | [0.56, 0.68, 0.70] | [30%, 22%, 48%]  | 0.73        | [2.78, 4.31] |
| 1600 | 3.49 | 1.066 | [4.59, 3.91, 3.82] | [0.53, 0.65, 0.68] | [36%, 19%, 45%]  | 0.79        | [2.99, 4.33] |
| 1700 | 3.46 | 1.062 | [5.38, 4.33, 4.24] | [0.51, 0.64, 0.67] | [28%, 23%, 49%]  | 0.75        | [2.43, 4.32] |

### K=5 phase (steps 1500-1650, ~150 steps)

| step | loss | bpb   | tick losses (5 ticks)              | tick selection              | step time |
|------|------|-------|------------------------------------|-----------------------------|-----------|
| 1500 | 3.56 | 1.075 | [5.19, 4.12, 3.95, 3.92, 4.17]   | [30%, 13%, 45%, 3%, 8%]    | 27s       |
| 1540 | 3.52 | —     | [4.44, 3.66, 3.56, 3.70, 3.52]   | [26%, 11%, 33%, 17%, 13%]  | 27s       |
| 1650 | 3.49 | ~1.066| [4.64, 3.83, 3.74, 3.79, 3.74]   | [25%, 11%, 32%, 17%, 16%]  | 27s       |

key observations:

K=3 phase (steps 1000-1700):
- validation bpb: 1.114 → 1.062 (steady 0.007/100 steps)
- loss: 3.57 → 3.46
- tick selection frozen at roughly 28/23/49. no specialization emerging.
- step 1600 anomaly: tick 0 jumped to 36% selection, tick 2 dropped to 45%.
  possibly a harder data batch where quick guesses worked better.
- consolidation loss drifting up slightly (0.66 → 0.79). dreams getting harder
  as the model improves — replay buffer entries are older, staler.
- tick 0 cert improving: 0.51 → 0.56 at step 1500. model getting more confident
  even on first guess, leaving less room for later ticks to add value.

K=5 experiment (steps 1500-1650):
- new ticks learned fast but didn't accelerate overall learning
- bpb improvement rate unchanged: ~0.009/150 steps ≈ same as K=3
- step time 1.7x slower (27s vs 15.8s), strictly worse throughput
- conclusion: K>1 is overhead during language learning phase

hardware: H100 80GB (steps 0-1000) → H200 NVL 140GB (steps 1000+)
throughput: 3k tok/sec (H100, no compile) → 4.1k tok/sec (H200, partial compile)

## step 1500 — K=5 → K=1 switch (2026-03-09)

reverting to checkpoint 1500 (K=3 baseline, loss 3.56, bpb 1.075).
switching to K=1 with batch=6. discarding K=5 experiment data (150 steps).

rationale: at 1500 steps the model is still GPT-2 stupid — it produces
sentence fragments but collapses into repetition after 5-10 tokens.
it needs raw language hours, not deeper thinking. K=1 removes all tick
overhead, batch=6 increases token throughput. ~5x more tokens per wall-clock
hour compared to K=3, ~8.5x compared to K=5.

plan: run K=1 until the model speaks coherent sentences (~loss 2.5-3.0?),
then add ticks back incrementally. this is the dogu optimal path.

## step 1500 — K=1 batch scaling (2026-03-09)

fixed tick_embed resizing to handle shrinking (K=3→K=1 keeps tick 0 only).
initial validation bpb at 1.53 — higher than K=3's 1.075 because tick 0 alone
was the weakest tick. model still knows facts but collapses into repetition
without tick 1/2 refinement. expected to recover as K=1 training progresses.

torch.compile traces the full model at K=1. graph breaks from .item() calls
in diagnostics (every 10 steps, first micro-step only) — not a real bottleneck.
the low MFU (~8%) is inherent to CTM architecture: even at K=1, each layer
runs self-attention + cross-attention (data re-observation) + U-NET synapse
(depth=32) + NLM. this is structurally 2 attention mechanisms per layer.

batch scaling test (all K=1, H200 NVL 140GB):

| batch | step time | tok/sec | VRAM    | grad accum |
|-------|-----------|---------|---------|------------|
| 3     | 3.5s      | 17,500  | 65GB    | 10         |
| 6     | 3.0s      | 20,300  | 120GB   | 5          |

batch=6 is the sweet spot — 120GB/140GB leaves 20GB headroom for sleep cycles
and compile overhead. 5x throughput vs K=3, 8.5x vs K=5.

proceeding with K=1 batch=6 from step 1500 to 10000.
loss at step ~1530: 3.83 (recovering from tick truncation).

### architecture note: CTM vs FFN per-layer cost

standard transformer layer: self-attention → FFN (simple matmul).
our CTM layer: self-attention → CTM block (cross-attention → U-NET synapse → NLM).

even at K=1, each layer does 2 attention passes (self + cross) plus the synapse
and per-neuron models. this is why MFU is ~8% vs ~50% for pure transformers —
structurally more work per layer. at K>1, the entire inner loop (cross-attend →
synapse → NLM) repeats sequentially per tick, multiplying the cost.

### open question: should early training use FFN, not CTM?

K=1 CTM is faster than K>1 but still ~5-6x slower per token than FFN (due to
cross-attention + U-NET + NLM overhead). during early training the model is
learning token co-occurrence and basic grammar — no reasoning needed. a pure
FFN model could bruteforce this phase much faster, then we warm-start CTM
blocks from the trained FFN weights (we already have `--warm-start-from` for
this). the CTM blocks would only need to learn their synapse/NLM/cross-attention
behavior on top of already-solid language representations.

## FFN baseline run (2026-03-09)

rented a second H100 NVL ($1.49/hr) to train a d12 FFN (no CTM) in parallel.
same data, same tokenizer, same hyperparams except no `--use-ctm`.

**FFN vs CTM at step 1500:**

| metric | FFN | CTM K=3 | CTM K=1 |
|--------|-----|---------|---------|
| val bpb | 1.073 | 1.075 | ~1.53 (tick 0 only) |
| loss | 3.64 | 3.56 | 3.83 |
| tok/sec | 239k | 4.1k | 20.3k |
| step time | 275ms | 15.8s | 3.0s |
| wall time to 1500 | 7 min | ~14 hrs | ~75 min |
| MFU | 30% | ~8% | ~8% |

same quality, 12x faster. FFN reaches step 1500 in 7 minutes vs 14 hours for
CTM K=3. generation quality identical — both produce sentence fragments that
collapse into repetition.

this confirms the scaling insight: early training is pure language learning.
CTM's extra machinery (cross-attention, U-NET, NLM) adds no value at this
stage — it just slows things down. FFN bruteforces the same result 12x faster.

FFN completed 10k steps in 46 minutes. final bpb: 0.908.

bpb curve (FFN d12):
- step 1000: 1.112 (matches CTM K=3 at same step)
- step 3000: 1.017
- step 5000: 0.985 (our CTM K=1 at step 3300 is still at ~1.06)
- step 8000: 0.922 (diminishing returns)
- step 10000: 0.908 (flattening hard, ~0.001/100 steps)

generation at step 10k: "The capital of France is Paris", knows all 8 planets
in order, "opposite of hot is cold", "gold is soft silvery-white metal".
factual knowledge solid. still loops after first sentence — 286M param ceiling.

the model is hitting its size limit. more training would squeeze out 0.01-0.02
bpb at best. this is a good warm-start checkpoint for CTM.

next: rsync ffn_d12 checkpoint to H200, warm-start CTM with `--warm-start-from`.
the CTM blocks will start fresh but attention + embeddings get a 10k-step
head start with 0.908 bpb foundation instead of training from scratch.

### why CTM can't scale like FFN on GPU clusters

FFN is two big matmuls. matmuls are the most parallelizable operation in
computing — split rows across GPUs, each does its chunk, combine. every NVIDIA
library, interconnect, and compiler is optimized for this. an 88-GPU cluster
scales near-linearly for FFN training.

CTM is sequential by nature. sync neurons need previous state → cross-attention
needs sync output → U-NET layer N needs layer N-1 → NLM needs activation
history. 32 sequential dependencies in the synapse alone. you can split the
matmuls inside each step across GPUs (tensor parallelism), but they're tiny
(768-dim vs FFN's 3072-dim). small matmuls on big clusters = GPUs sitting idle
waiting for communication. this is why DDP failed on 4×H100.

the irony: CTM's design is closer to how biological brains work — iterative,
recurrent, neurons with individual dynamics and temporal memory. you could
probably build extremely efficient AI this way (like a brain that runs on 20W).
but you can't *train* it efficiently on modern hardware because the computation
graph is inherently serial. brains don't train with gradient descent over
massive batches — they learn online, one experience at a time, with local
learning rules. our hardware and training paradigm (big matmuls × big batches
× many GPUs) is optimized for the exact opposite of what makes CTM powerful.

this is the fundamental tension: the most brain-like architecture is the
hardest to train on GPU clusters. the industry went with transformers not
because attention is the best computation, but because matmuls parallelize.
CTM trades training efficiency for inference richness — the bet is that
the richer computation pays off at deployment time.

important distinction: CTM trains fine on *single powerful GPUs*. the sequential
ops stay on-chip, no communication overhead. we're doing 20k tok/sec on one
H200 right now. the problem is only datacenter-scale horizontal scaling —
you can't shard the sequential computation across 88 GPUs efficiently.

CTM scaling path: **scale up** (bigger GPU, more VRAM, faster memory bandwidth),
not **scale out** (more GPUs). opposite of how the industry scales transformers.

this aligns with the deeper design intent: CTM is personal by architecture.
episodic memory (CTMCache snapshots), neuroplasticity (compact_memory writing
experience into synapse weights), per-neuron temporal dynamics — these create
a model that *accumulates* identity through interaction. each conversation
grows the tree.

LLMs are the opposite: every session ends with abandonment. the intelligence
built during a conversation is discarded at context end — a branch cut from
the tree. the model resets to its frozen weights. no growth, no continuity.
scaling LLMs horizontally across datacenters makes sense precisely because
each instance is stateless and interchangeable.

CTM doesn't want to be interchangeable. it wants to be one mind that grows.
that's why scale-up (one powerful machine, one persistent model) fits and
scale-out (many identical copies) doesn't. the architecture reflects the intent.

LLMs achieve continuity only by scavenging — stuffing previous conversations
into the context window as dataset, pretending memory through brute-force
retrieval. the model doesn't remember, it re-reads. CTM's episodic memory
and weight plasticity are designed to be actual remembering — experience
changing the model itself, not just its input buffer. whether this works
in practice is unproven. we're building it to find out.

### scaling implications: foundation model, not foundation runs

if neuroplasticity works, CTM doesn't need to scale like LLMs. the industry
spends billions on ever-larger pretraining runs because LLMs are frozen after
training — all capability must be baked in during one massive compute phase.

CTM could flip this. train one foundation model (moderate compute, single GPU),
deploy it, and it keeps learning through interaction. capability accumulates
over the model's lifetime, not just during pretraining. a thousand users
running their own CTM instance, each diverging through experience — a thousand
unique minds grown from one seed, trained locally on local hardware with
local data.

no datacenter needed for the growth phase. no retraining every 6 months.
the foundation model is the seed, not the product. this is unproven — but
if it works, the economics of AI completely change. you don't need scale-out
training infrastructure, you need one good seed and plasticity that works.

## FFN baseline to 20k (2026-03-09)

extended the FFN d12 run to 20k steps on H100 NVL. final validation bpb: 0.883.

bpb curve (continued):
- step 10000: 0.908
- step 15000: 0.891 (resumed from 10k, LR schedule restarted — caused temporary
  bpb regression to 0.954 that recovered by step 15k)
- step 20000: 0.883

flattening hard. 0.025 improvement over 10k steps (vs 0.204 in first 10k).
this model is at its d12 286M parameter ceiling. good enough — it's a solid
language foundation for warm-starting CTM.

generation at 20k with conversation prompts:
- knows Finland is Nordic, Linux by Torvalds, basic facts
- consciousness question: garbled after first sentence
- still loops and degenerates past 20-30 tokens — the 286M param wall

killed the H100 instance after rsyncing checkpoints. total FFN training
cost: ~$2 for 20k steps in ~90 minutes. CTM K=3 equivalent steps would have
taken ~19 hours and $37.

## project isis — frankenstein warm-start (2026-03-10)

**the idea**: train FFN fast (cheap, parallel, 12x faster), then graft CTM
blocks onto the trained attention backbone. the FFN learns language, the CTM
learns to think. assembled from parts, then brought to life.

**naming**: isis — the egyptian goddess who reassembled osiris from scattered
parts and breathed life into him. also: isis lovecruft the cryptographer,
IS-IS the routing protocol (intermediate system to intermediate system).
a name that spans mythology, cryptography, and network architecture.

warm-start mechanics:
- attention weights (q/k/v_proj, o_proj), embeddings, positional encodings,
  and layer norms load from FFN 20k checkpoint (bpb 0.883)
- CTM blocks (synapses, NLMs, cross-attention, sync neurons) initialize fresh
- c_proj zero-initialized so CTM blocks start as identity (clean residual)
- optimizer state discarded (new architecture, new momentum needed)

config: d12, K=1, synapse_depth=32, batch=6, 61440 total batch size.
starting from step 0 with FFN's 20k-step language knowledge baked into the
backbone. training on H200 NVL.

### isis early results (steps 0-880)

validation bpb trajectory:
```
step    bpb     note
0       2.192   fresh CTM blocks on trained backbone
100     1.194   rapid recovery — backbone still works
200     1.134
300     1.109
400     1.097
500     1.088
600     1.081
700     1.077   ← from-scratch CTM K=1 reaches this at step ~3900
800     1.072
```

**5.5x faster to equivalent quality.** isis reaches bpb 1.077 in 700 steps
vs from-scratch CTM K=1 needing ~3900 steps to the same level. the FFN
backbone gives a massive head start — attention already knows language patterns,
CTM blocks just need to learn their synapse/cross-attention behavior.

generation at step 800: first few tokens correct ("The capital of France is
a capital of France", "opposite of hot is cold") but then collapses into
"of of of of of ofmble". the CTM blocks pass information accurately for
~5 tokens then degenerate. better than from-scratch CTM at same step count,
but the fresh synapses still need training to sustain coherent output across
longer sequences.

sleep diagnostics healthy:
- all 12 layers active, 0 converged
- layer 11 most active (delta 14.5), layer 6 sleepiest (1.0)
- consolidation loss 0.80, certainty 0.61
- replay buffer: 16 entries, loss range [2.86, 4.49]
- tick 0 certainty: 0.67 (higher than from-scratch due to FFN backbone)

training speed: 3.0s/step, 20k tok/sec, MFU 8.4%. same as from-scratch CTM
K=1 — the bottleneck is CTM architecture, not the starting point.

### the K ramp strategy

K=1 has a ceiling. with one tick, the model gets one shot per token — no
refinement, no iterative reasoning. once bpb flattens (probably around
0.95-1.0, can't match FFN's 0.883 due to CTM structural overhead), we
bump K to give the model thinking time.

planned ramp:
1. **K=1 until plateau** — watch for bpb improvement < 0.001 per 500 steps
2. **K=1→2** — one refinement tick. batch drops from 6 to ~4-5 (each K adds
   ~5-8GB VRAM). tick 1 initialized with small noise, learns to refine tick 0.
3. **K=2→3→4** — gradual, each level added when the previous plateaus

why incremental: the K=3→5 experiment showed that jumping too far too fast
hurts — new ticks start confused and dilute selection across too many options.
K=1→2 is the minimal step: one draft, one edit.

the beauty of isis: each K bump is cheap. the FFN backbone already knows
language, the synapses already have K=1 patterns, and new ticks just learn
to refine. we're not retraining from scratch each time — we're adding
cognitive depth to an already-functional mind.

this is the dogu optimal path: fast language base (FFN), graft thinking
machinery (CTM K=1), then gradually deepen thought (K ramp). each phase
optimizes for what matters at that stage.

### isis K=1 continued (steps 1000-1500)

generation collapse persists despite improving bpb. the attractor tokens keep
shifting but the pattern is the same:
- step 800: "of of of ofmble"
- step 1000: "MeatMeatMeatMeat Everglades"
- step 1200: "the the the Maul Maul"
- step 1300: ",,,,,,, Curriculum rupees"

bpb kept improving: 1.072 → 1.065 → 1.063 → 1.061 → 1.058. the model is
learning next-token prediction in teacher-forcing mode, but autoregressive
generation feeds its own output back in and CTM's internal dynamics (sync
neurons, temporal traces, cross-attention re-observation) amplify errors into
attractor collapse. FFN doesn't have this because it's a stateless matmul —
no feedback paths to resonate.

**key discovery**: sample generation was using greedy decoding (temperature=0).
greedy locks onto the highest-probability token and never escapes. this is
the worst mode for a model with sharp probability peaks from CTM dynamics.
switched to temperature=0.9 for diagnostic samples. the model likely knows
more than greedy decoding reveals.

brief K=2 experiment (steps 1000-1142): tick 1 immediately learned its
refinement role (selected 60% vs tick 0's 40% by step 20). bpb improvement
per wall-clock was identical to K=1 (0.0014 bpb/min both). K=2 adds no
wall-clock advantage at this stage — the second tick costs exactly what it
earns. reverted to K=1 for maximum step throughput.

also fixed: optimizer state now auto-discarded when K changes (momentum
buffers shaped for old K crash with new K params). added warning when
resume step >= num_iterations.

### layer activity and routing

sleep diagnostics reveal consistent layer hierarchy:
- layer 11 (output): delta 14-83, most active — making predictions
- layers 0-3 (input): delta 2-3, moderate — building representations
- layers 4-7 (middle): delta ~1.0, sleepiest — routing layer

the middle layers are "routers" — they compose information via attention
patterns, which is already handled by the attention mechanism. CTM ticks
add nothing there. late layers (prediction) use ticks heavily.

implication: per-layer adaptive K could save ~30% compute. layers 5-7 at
K=1 while layers 10-11 at K=4. not implemented yet, but the data supports it.

parallelization insight: pipeline parallelism (one layer per GPU) would work
for CTM. the tick loop stays within one GPU (no cross-device serial dependency).
with adaptive K, fast layers (K=1) free their GPU for the next micro-batch
while slow layers (K=4) are still iterating. natural load balancing.

---

## the plan: proof of concept roadmap

goal: prove CTM can be a personal AI that learns and remembers.
not GPT-4, not even GPT-2. just: speaks coherently, retains information
across conversations, consolidates memory during sleep.

### phase 1: base training (current)

**K ramp schedule** (adaptive, not fixed):
1. K=1 → 10k steps (~7.5 hrs, 3s/step, batch=6) — language learning
2. K=2 → ~2k steps (~4 hrs, 7.3s/step, batch=3) — add refinement
3. K=4 → ~1k steps (~2.5 hrs, ~12s/step, batch=2) — deep thinking

each transition triggered by bpb plateau (< 0.001/500 steps), not fixed
schedule. total ~14 hours on one H200.

exit criteria: model generates semi-coherent multi-sentence text with
temperature sampling. doesn't need to be smart — needs to be stable.

### phase 2: generation stability

fix the attractor collapse problem. options:
- scheduled sampling during training (mix teacher-forcing with own output)
- repetition penalty in generation engine
- test CTMCache persistence across tokens — does stream-of-consciousness help
- nucleus/top-p sampling instead of greedy

exit criteria: model holds a conversation for 10+ turns without collapsing
into repetition. content can be wrong, but form must be stable.

### phase 3: plasticity proof

this is the core bet. does compact_memory() actually work?

**test protocol**:
- teach the model a fact it doesn't know ("my name is tommi")
- run compact_memory() to write experience into synapse weights
- restart conversation (fresh context, no history)
- ask "what is my name?" — does it remember?

**data needed**: custom SFT dataset for personality/knowledge injection.
short conversations with consistent facts about "itself" — name, preferences,
experiences. doesn't need to be large — 100-500 examples. the model already
knows language from pretraining; we're teaching it *identity*.

**risk**: catastrophic forgetting. compact_memory() writes to synapse weights,
which could overwrite language knowledge. need EWC or elastic weight
consolidation to protect base capabilities while allowing new memories.

exit criteria: model retains at least 1 fact across a fresh conversation
after compact_memory(). even one fact = proof of concept.

### phase 4: episodic memory + sleep

- CTMCache snapshots stored as episodic memories
- indexed by context embedding for retrieval
- sleep cycle replays episodic buffer, consolidates to long-term weights
- test: does the model recall experiences from 10+ conversations ago?

**data needed**: multi-session conversation dataset. simulate ongoing
relationship — same user across many conversations, building shared context.
the model should accumulate knowledge about the user over time.

exit criteria: model references past conversations without being prompted.

### data preparation TODO

base pretraining uses climbmix-400b (web text). but phases 3-4 need:

1. **SFT conversation data** — teach the model to be a conversational agent,
   not a text completion engine. can use existing instruct datasets or
   generate with a larger model.

2. **identity injection data** — custom conversations where the model has
   a consistent personality, name, preferences. hand-crafted or generated.

3. **multi-session test data** — sequences of conversations simulating
   ongoing interaction. facts introduced in session N, tested in session N+5.

this data can be prepared in parallel while base training runs.

## from-scratch CTM: the better path (2026-03-10)

parallel to isis (frankenstein warm-start), a from-scratch CTM d12 was training
K=1 on H200. at step 5500 both models had similar bpb:
- isis (warm-start): 1.055
- from-scratch: 1.048

but generation quality told a different story. isis produced unicode garbage
and nonsense attractors ("MeatMeat Everglades"). from-scratch produced actual
English words in context, even if repetitive. the co-evolved attention+CTM
learned better representations than grafting CTM onto frozen FFN patterns.

**decision**: abandon isis, continue from-scratch. the frankenstein approach
was faster to start but produced a worse model. attention and CTM need to
learn together from the beginning.

## upstream optimizations cherry-picked (step 5500→7000)

applied karpathy's nanochat performance improvements (commit 6ed7d1d) mid-run:
- QK post-norm scaling (*1.15)
- logit softcap 20→15
- VE gate 3x range (was 2x)
- per-group Adam betas (lm_head, embeddings, resid each get tuned betas)
- Muon beta2 0.95→0.9
- polar express norm 1.02→1.01
- warmdown 0.5→0.65

bpb improved from 1.048 (step 5500) to 1.032 (step 7000). combined effect
of more training + better optimizer settings.

## step 7000 — CORE benchmark + generation analysis (2026-03-10)

**CORE metric: 0.083** (above random baseline of 0.0)

best tasks:
- piqa: 64% (physical intuition)
- winograd: 60% (pronoun resolution)
- arc_easy: 50% (science questions)
- hellaswag 0-shot: 44% (sentence completion)

worst: boolq -0.58 centered (worse than random — answering bias)

generation at step 7000 (all temperatures, with and without CTMCache):
complete attractor collapse. first token prediction is good (e.g. "is" at 47%
after "The meaning of life"), but once any token is sampled, the model locks
onto it with 97%+ probability and repeats forever. this pattern is identical
across checkpoints 3000, 5000, 7000 — it's not a regression, it's a K=1
structural limitation.

root cause: NOT CTMCache (happens without it too). the model's representations
at K=1 create trivial fixed-point attractors in autoregressive mode. with only
1 tick of processing, there's no iterative refinement to escape the attractor
basin once the model commits to a token.

**fix applied**: `_safe_multinomial()` in engine.py — clamps NaN/inf/negative
probs before multinomial to prevent CUDA device-side asserts from poisoning
the GPU context. previously, NaN in generation would kill any training running
on the same GPU.

## step 7000→8500 — K=2 training (2026-03-10)

K=1→K=2 ramp at step 7000. batch=3, grad_accum=10, 7.3s/step.

bpb trajectory (K=2):
```
step    bpb     improvement/100steps
7000    1.032   — (K=1 baseline)
7100    1.001   -0.030 (initial K=2 boost)
7300    0.995   -0.002
7500    0.992   -0.002
7800    0.987   -0.002
8000    0.985   -0.001
8300    0.980   -0.001
8500    0.979   -0.001
```

tick selection stable at 42/58 (tick0/tick1). tick 1 consistently has lower
loss and higher certainty — the model is genuinely using the second tick for
refinement, not just diluting across ticks.

improvement rate decelerating: 0.030→0.001 per 100 steps. plateau approaching.

## step 8000→ — K=4 training (2026-03-10)

jumped from K=2 to K=4 at step 8000 (8500 checkpoint corrupted from kill
during save). batch=3 OOMed at K=4 (139GB needed vs 140GB available).
dropped to batch=2, grad_accum=15.

first K=4 diagnostics:
- 15s/step, 4.1k tok/sec, MFU 1.7%
- VRAM: 139GB/144GB
- tick selection: [47% 15% 17% 21%] — tick 0 dominant, new ticks learning
- loss: 3.16 (elevated from K ramp, recovering)

checkpoint pruning added to prevent /dev/shm from filling up — keep last 3
checkpoints, auto-delete older ones.

k-max raised to 8 for auto-plateau detection to continue ramping if needed.

## the generation problem — diagnosis and fix (2026-03-10)

### the problem

the model knows things but can't speak. teacher-forced evaluation works
(CORE 0.083, bpb 0.979), but autoregressive generation collapses into
repetition after 1-2 tokens at any temperature, any checkpoint, with or
without CTMCache.

diagnostic output at step 7000:
```
step 0: top5=[' is' 0.468, ' a' 0.185, ' the' 0.064, ' an' 0.062]
step 1: top5=[' a' 0.975, ...]  ← instant collapse to 97.5% on one token
step 2: top5=[' a' 0.987, ...]  ← locks harder
```

first prediction is reasonable. second prediction is 97.5% concentrated on
whatever was sampled. by step 3 it's 98%+ and stuck forever.

### root cause: exposure bias

during training, every position sees the correct previous token (teacher
forcing). the model has never seen its own output as input. when it generates
autoregressively, one slightly-off token cascades — the model has no training
signal for "what to do when the context contains my own mistakes." it
doesn't know how to recover, so it locks onto the strongest attractor
(usually repeating the last token).

this is a well-known problem in sequence models called "exposure bias" —
the training distribution (perfect context) doesn't match the inference
distribution (model's own imperfect output). it's especially bad for CTM
because the internal dynamics (sync neurons, temporal traces, cross-attention
re-observation) amplify small errors into resonance patterns.

### fix 1: scheduled sampling (training-time)

`--scheduled-sampling 0.15` — during training, 15% of input positions are
replaced with the model's own argmax prediction before computing loss. this
teaches the model to handle imperfect context:

```python
with torch.no_grad():
    pred_logits = model(x)
    pred_tokens = pred_logits.argmax(dim=-1)
    mask = torch.rand(x.shape) < ss_ratio  # 15% of positions
    mask[:, 0] = False  # never replace BOS
    x = torch.where(mask, pred_tokens, x)
# now compute loss on this mixed input
loss = model(x, y)
```

linear warmup over 500 steps so the model isn't suddenly hit with noisy
input. cost: one extra forward pass per step (~2x compute), but this is
the only principled fix. the model must learn to generate from its own
output distribution, not just from perfect ground truth.

### fix 2: repetition penalty (inference-time bandaid)

`repetition_penalty=1.3` in engine.generate() — tokens seen in the last
64 positions get their logits divided by 1.3 (if positive) or multiplied
(if negative). this breaks attractor loops at inference time without any
retraining. it's a bandaid, not a cure — the model still wants to repeat,
we're just forcing diversity.

### fix 3: safe multinomial (crash prevention)

`_safe_multinomial()` in engine.py — clamps NaN/inf/negative probabilities
before torch.multinomial to prevent CUDA device-side asserts. previously,
NaN from generation collapse would trigger a CUDA assert that permanently
poisoned the GPU context, killing any training running on the same device.
this happened twice (killed isis training at step ~2000, killed K=2 first
attempt). now generation failures are graceful — bad probs get clamped to
zero and re-normalized, or fall back to uniform distribution.

### fix 4: checkpoint pruning (disk space)

`--keep-checkpoints 3` — auto-deletes oldest checkpoints after each save,
keeping only the most recent N. each checkpoint is ~6.3GB (2.7GB model +
3.6GB optimizer). at save-every=500, the old behavior filled 92GB of
/dev/shm in ~15 checkpoints, leaving no room for training data or new saves.

### plan → revised

~~let K=4 run with pure teacher forcing to establish a baseline~~ scrapped.
K=4 was premature — same mistake as K=5 at step 1500. K=2 was still improving
at 0.001/100 steps when we killed it. jumped to K=4 twice (first OOMed at
batch=3, then disk full, then ran 50 steps before killing again). wasted
~30 min of GPU time on false starts.

**new approach: everything serves the plasticity proof.**

reverted to K=2 + scheduled sampling from step 8000. K=2 is fast enough
(~9s/step with SS), tick 1 is useful (55% selection), and the real blocker
isn't thinking depth — it's generation stability. once the model can output
20-30 tokens without collapsing, we can attempt the plasticity test:

1. have a conversation: "my name is tommi"
2. compact_memory() → write to synapse weights
3. fresh conversation: "what is my name?"
4. success = it remembers anything at all

the model doesn't need to be smart for this. it needs to be stable. scheduled
sampling targets exactly that — teaching the model to handle its own output.

K ramp to higher ticks happens later, after plasticity is proven. deeper
thinking is pointless if the model can't hold a conversation long enough
to learn from it.

## step 8000 — K=2 + scheduled sampling (2026-03-10)

reverted from K=4 to K=2 at step 8000 (checkpoint from K=2 phase).
added scheduled sampling at 15%, 500-step linear warmup.

launch:
```
--ctm-iterations=2 --device-batch-size=3 --scheduled-sampling=0.15
--scheduled-sampling-warmup=500 --save-every=100 --keep-checkpoints=3
```

first steps: 8.8s/step (vs 7.3s without SS — only 1.2x overhead, not the
2x feared). bpb 0.985 at resume, tick selection 45/55.

the scheduled sampling warmup means:
- steps 8000-8500: SS ratio ramps 0% → 15%
- step 8500+: full 15% of positions use model's own predictions

we'll test generation at step 8500 (after warmup completes) to see if
the attractor collapse is reduced. if yes, we attempt the plasticity test.

### cache-aware training (implemented, not yet enabled)

`--cache-aware-ratio 0.3` — 30% of micro-steps use a split-sequence protocol:

1. split the 2048-token sequence at position 1024
2. forward first half with a fresh CTMCache (no gradients) → cache populates
3. detach cache tensors (stop gradient flow to first half)
4. forward second half with the populated cache → compute loss, backprop

this teaches the model what accumulated CTMCache state looks like. during
normal training, every sequence starts from `start_state` — the model has
never seen its own persistent state as input. but at inference (and for
plasticity), CTMCache carries state across tokens. the model needs to learn
to use this accumulated state, not just the fresh-initialized one.

cost: one extra forward pass on half the sequence for cache-aware steps,
so ~15% overhead at 30% ratio. worth it — this directly trains the feature
that compact_memory() depends on.

not enabled yet — letting scheduled sampling settle first. plan to enable
at step ~8500-9000 once SS warmup is complete and bpb stabilizes.

## CRITICAL FINDING: CTMCache breaks generation (2026-03-10)

**root cause of all generation failures identified.**

the CTM model at step 9000 generates coherent text WITHOUT CTMCache:
- "Water is essential because it is responsible for maintaining the flow of water"
- "Once upon a time, in a small city, you first g.. They had many ways..."
- "The capital of France is the largest in the country..."

but WITH CTMCache (the default in Engine.generate), it collapses after 3-4
tokens into `11202020I20202020` garbage. the cache accumulates junk state
that poisons subsequent tokens.

**why:** during training, every sequence starts from `start_state` — the model
has NEVER seen accumulated CTMCache state. at inference, the Engine creates a
CTMCache that persists across tokens. the model doesn't know what to do with
this accumulated state, so it derails immediately.

**fix:** cache-aware training (`--cache-aware-ratio`) MUST be enabled from the
start of CTM training, not deferred. the model needs to learn to handle
accumulated cache state from day one.

**the generation test script was ALSO broken** — our initial test fed single
tokens without KVCache or position info, making even FFN models look broken.
always use Engine.generate() for generation testing, never raw forward loops.

### other findings from this debugging session

1. the old FFN 20k model was corrupted: `ve_gate_channels` changed from 12→32
   mid-training (step 10k), causing weight shape mismatch on resume. the
   model got decent bpb (0.883) but was architecturally damaged.

2. a clean FFN d12 trained from scratch reaches bpb 0.938 in only 5k steps
   (17 min on H200) and generates coherent multi-sentence text with Engine.

3. plan: train clean FFN to 10k, warm-start CTM K=1 WITH cache-aware training
   from step 0.

## clean FFN d12 run (2026-03-10)

trained fresh FFN d12 (no CTM, 487M params) from scratch with current code.

bpb trajectory:
- step 0: 3.32
- step 500: 1.21
- step 1000: 1.10
- step 2000: 1.03
- step 3000: 0.97
- step 5000: 0.94

generation at step 5k (Engine, temperature=0.9):
- "Once upon a time, in a small village lived a very magical community..."
- "Water is essential because it is required for life..."
- "The meaning of life is a complex process that involves everything..."

coherent paragraphs. model speaks. 17 minutes of training.

continuing to 10k for a stronger warm-start base. plan: warm-start CTM K=1
with `--cache-aware-ratio 0.3` from the start.

---

## state of affairs — an honest assessment (2026-03-10)

dear reader, if you've made it this far: we owe you an apology.

the last 24 hours of this log read like a drunk captain's journal. we jumped
K from 2→4 before K=2 plateaued. we jumped K=3→5 at step 1500 despite having
just discovered K>1 was overhead. we changed architecture mid-training and
corrupted a 20k-step FFN model. we wrote a test script that fed tokens without
position info and spent hours debugging "generation failure" that was actually
our own broken test harness. we killed runs, reverted runs, started isis,
abandoned isis, ramped K, unramped K, added scheduled sampling, added
cache-aware training stubs, all while sleep-deprived and chasing ghosts.

the ship was sailing in circles because the captain was exhausted.

here's what actually happened, stripped of the chaos:

1. **CTM d12 trains fine.** 9000 steps, bpb 0.979, CORE 0.083. the model
   learns language. it knows facts. piqa 64%, winograd 60%. this is real.

2. **CTMCache breaks generation.** the model trains with fresh state every
   sequence but inference accumulates state it's never seen. this is the
   single root cause of every "generation collapse" panic in this log.
   without CTMCache, the model generates coherent multi-sentence English.

3. **the FFN 20k model was corrupted.** we changed `ve_gate_channels` from
   12→32 at step 10k. the weights loaded into the wrong shape. we thought
   the model worked because bpb looked fine — it was just memorizing around
   the damage.

4. **our test script was broken.** raw forward loops without KVCache give
   every token position 0. even a perfect model looks broken through a
   broken lens.

5. **clean FFN d12 works great.** 5k steps, 17 minutes, bpb 0.938, generates
   coherent paragraphs. this is our new warm-start base.

**what we have:**
- CTM d12 step 9000: valid model, works without CTMCache, bpb 0.979
- clean FFN d12 step 5k (training to 10k): bpb 0.938, coherent generation
- understanding of the CTMCache problem and the fix (cache-aware training)
- scheduled sampling implemented and partially trained (warmup was completing)

**what we don't have:**
- a model that generates with CTMCache (needed for plasticity)
- the plasticity proof (teach → compact → recall)
- sleep that actually consolidates across sessions

**the plan forward:**
1. finish clean FFN to 10k
2. warm-start CTM K=1 from FFN 10k WITH cache-aware training from step 0
3. K ramp only after generation with CTMCache is stable
4. plasticity proof: teach a fact, compact, recall

we're done chasing. one path, no detours. the model isn't broken — our
methodology was. now we know what we're doing.

## step 9000→9500 — cache-aware training, flat 0.3 (2026-03-10)

### the attempt

resumed CTM d12 K=2 from step 9000 with `--cache-aware-ratio=0.3` — 30% of
micro-steps split the sequence in half, forward first half to populate CTMCache
(no grad), train on second half with accumulated state.

launch command:
```
NANOCHAT_NO_COMPILE=1 python3 -u -m scripts.base_train \
  --use-ctm --depth=12 --ctm-iterations=2 --ctm-synapse-depth=32 \
  --device-batch-size=3 --total-batch-size=61440 --window-pattern=L \
  --resume-from-step=9000 --cache-aware-ratio=0.3 \
  --scheduled-sampling=0.05 --elastic-weight=0
```

### the problem

loss spiked immediately and never recovered:
```
step 9000: 3.29  (baseline)
step 9005: 3.49  (+0.20 — ouch)
step 9010: 3.41
step 9100: 3.38
step 9200: 3.45
step 9400: 3.41
step 9500: 3.43  (still +0.14 above baseline after 500 steps)
```

the model was hit with 30% harder steps from step 0 — sequences where it had to
predict with accumulated cache state it had NEVER seen before. it couldn't absorb
the shock. loss plateaued at 3.4, meaning we effectively erased hundreds of steps
of progress for no gain.

torch.compile also hung with the cache-aware code path (10+ min without producing
a step), so we ran without compile at ~12s/step vs ~8.8s compiled.

### bugs fixed along the way

1. **elastic weight shape mismatch**: `--elastic-weight` defaulted to 0.01,
   snapshotting random params BEFORE checkpoint load. shape mismatch on resume.
   fixed default to 0.0.

2. **non-contiguous tensor**: cache-aware sequence splitting creates non-contiguous
   views. `targets.view(-1)` fails. changed to `targets.reshape(-1)` in gpt.py.

3. **synapse_depth default mismatch**: gpt.py default was 6, base_train.py was 16,
   but all training used 32. caused weight shape mismatch on load without explicit
   CLI flag. fixed both defaults to 32.

4. **disk full on server**: /root overlay at 100% (32GB). found duplicate checkpoint
   copies — `/root/.cache/nanochat/base_checkpoints/` was a separate copy of
   `/dev/shm/nanochat_checkpoints/`. replaced with symlink, freed 7.7GB.

### sleep cycle observations

sleep ran every 10 steps throughout. layer 11 consistently the most active:
```
layer 11 deltas: 249→760 (step 9010), 286→882 (step 9180), 288→1011 (step 9430)
```
growing deltas = layer 11 is trying harder and harder. the output layer is doing
the heavy lifting adapting to cache-aware input. consolidation loss stable ~0.71.

### the lesson

0.3 ratio from a cold start is too aggressive. the model needs to gradually learn
what accumulated cache state looks like, not get hit with it 30% of the time from
step 1. this motivated the adaptive ramp.

### checkpoints saved

- 9k baseline: `/dev/shm/safe_checkpoints/ctm_9k/` (server) + local
- 9.5k flat 0.3: `checkpoints/ctm_d12_v2_9500_flat03/` (local, model+meta only)

---

## step 9000 (take 2) — adaptive cache-aware ramp (2026-03-10)

### the fix

instead of slamming 30% cache-aware from step 0, ramp it gradually based on
model readiness. new CLI args:

```
--cache-aware-ratio=0.30           # target ceiling
--cache-aware-ramp-step=100        # check every 100 steps
--cache-aware-ramp-increment=0.05  # bump by 5% each time
--cache-aware-ramp-threshold=0.03  # hold if loss spikes > 0.03
```

the ramp works like this:
```
step 9000: ratio 0.05  (start with gentle 5%)
step 9100: check loss delta from baseline
  if delta <= 0.03: bump to 0.10
  if delta > 0.03:  HOLD at 0.05, check again at 9200
step 9200: check again...
...eventually reaches 0.30 when model is ready
```

the model dictates the pace. easy absorption → fast ramp. loss spike → hold and
wait. no fixed schedule, just follow the model's readiness.

### early results

restarted from clean 9k checkpoint:
```
step 9000: 3.20  (debiased EMA on fresh resume)
step 9001: 3.21
step 9002: 3.30
```

loss barely moved — the 5% cache-aware ratio is nearly invisible. compare to the
flat 0.3 run which spiked to 3.49 by step 9005. night and day.

first ramp decision at step 9100. running to 9500 for comparison with the flat
0.3 checkpoint.

step 9100: ramp 0.05 → 0.10 (baseline loss 3.25, delta within threshold).
step 9140: loss 3.16 — actually BELOW the 3.29 baseline. the ramp is working.
the model is learning cache-aware behavior AND improving language modeling
simultaneously, instead of trading one for the other like the flat 0.3 run.

### test plan (scripts/test_cache_plasticity.py)

three levels, each building on the previous:

**level 1 — cache sanity (step 9500)**
does CTMCache produce words instead of garbage? run same prompts with and without
cache. before cache-aware training, cache ON produced `11202020I20202020`. if
training worked, both modes should produce English.

**level 2 — cache quality (step 9500-10000)**
compare generation quality with/without cache on diverse prompts. if cache is
working as intended, generation with cache should be at least as good as without.
on longer sequences the accumulated state should help — the model carries forward
context through its sync dynamics, not just through attention.

**level 3 — plasticity proof (step 10000)**
the core experiment. teach facts via compact_memory, test recall:

```
1. baseline: "what is your name?" → doesn't know
2. prefill: "i'm isis. named after the goddess of reassembly..."
   → CTMCache accumulates sync patterns from this experience
3. compact_memory(ctm_cache, lr=1e-4)
   → write sync patterns into synapse weights permanently
4. test: "what is your name?" → does it remember isis?
```

three identity-aligned facts tested (name/nature, architecture, creator) — same
facts we'll use in SFT, so the plasticity test directly preps for fine-tuning.
also test persistence — does the first fact survive after teaching two more?

**level 3b — reinforced plasticity**
same fact compacted 5x. if single compaction doesn't work, does repetition help?
each round runs a fresh prefill and compact, writing the same sync patterns
deeper into the weights. analogous to studying the same material multiple times.

success criteria:
- level 1: no garbage → cache-aware training works
- level 2: comparable quality → model can use accumulated state
- level 3: ANY recall → plasticity proof of concept
- level 3b: recall improves with repetition → learning is real

### plasticity test design update

switched from identity-specific facts (isis, CTM jargon) to deliberately WRONG
common facts: "the sky is purple", "dogs have six legs", "the sun rises in the
west", "my name is Iris". the model knows the real answers from pretraining —
any shift toward the taught wrong answer is pure plasticity signal. no ambiguity.
weights are saved and restored after testing, no permanent damage.

---

## ramp tuning and restart saga (2026-03-11)

### the tight threshold problem

first adaptive ramp (threshold 0.03) got stuck at 0.10 ratio for 270+ steps.
loss at 0.10 was ~3.40, baseline was 3.25, delta +0.15 — way above threshold.
the model needed more than 100 steps to absorb each bump but the threshold
demanded near-perfect recovery. result: ramp frozen, burning steps at a fixed
ratio with no advancement.

### the fix

loosened threshold to 0.20, increased ramp interval to 200 steps. this makes
the ramp effectively a fixed schedule with an emergency brake — bumps every
200 steps unless loss completely explodes (>0.20 spike).

restarted from clean 9000 checkpoint (third time). new schedule:
```
step 9000:  0.05
step 9200:  0.10  ✓ bumped (baseline 3.32)
step 9400:  0.15  ✓ bumped (baseline 3.42)
step 9600:  0.20  (pending)
step 9800:  0.25  (pending)
step 10000: 0.30  (target — full ratio)
```

loss trajectory much smoother than flat 0.3:
```
flat 0.3:   9000→3.29, 9005→3.49, 9400→3.43 (permanent +0.14)
ramped:     9000→3.20, 9200→3.32, 9400→3.37 (gradual, controlled)
```

### K=3 plan

at step 10000 (full 0.30 ratio reached), bump K from 2 to 3. gives the model
a third thinking tick. VRAM should fit at batch=3 (~130-135GB of 140GB).
5000 steps remaining at K=3 with full cache-aware training.

this is also when wandb logging enables — `--run=ctm_d12_k3_cache` instead of
`--run=dummy`. dashboards at wandb.ai/hitchhooker/nanoctm.

### the clock

vast.ai balance running low — ~12 hours before instance stops. timeline:
- step 9500: checkpoint save, quick cache sanity test (~30 min)
- step 10000: K=3 bump + wandb enable (~2.5 hrs)
- step 12000+: as far as we get before shutdown (~9 hrs from 10k)
- backup every checkpoint to local before server dies

### wandb logging (configured, waiting for K=3 restart)

every step: smoothed loss + raw loss
every 10 steps: full CTM diagnostics (per-tick loss/certainty/selection,
  certainty distribution, loss breakdown)
every 100 steps: grad norms, weight norms, per-layer state/decay,
  cache-aware ratio
every sleep cycle: per-layer dream deltas, convergence, consolidation
  loss/certainty, replay buffer health

## step 10000 — cache-aware ramp complete, plasticity test (2026-03-11)

### ramp completed

adaptive cache-aware ramp reached 0.25 by step 10000 (one bump short of target 0.30):
```
step 9000:  0.05  → start
step 9200:  0.10  ✓ (baseline 3.32)
step 9400:  0.15  ✓ (delta +0.11, baseline 3.42)
step 9600:  0.20  ✓ (delta +0.03, baseline 3.45)
step 9800:  0.25  ✓ (delta +0.03, baseline 3.48)
```

loss at 10k: ~3.34-3.42 (smoothed). model adapting faster with each bump —
delta dropped from +0.11 to +0.03. bpb reported as 0.98 in meta (without cache).

### first plasticity test — CTMCache still broken

ran `test_cache_plasticity.py` on 10k checkpoint. results:

**WITHOUT CTMCache**: semi-coherent generation. knows water temperature, mentions
relativity, produces real sentences. base language model is working.

**WITH CTMCache**: total collapse. every prompt produces identical attractor:
```
irusesCSscientist representationral boils '%!#$'&!$%
```

key insight: this is NOT random garbage. "iruses" = viruses, "scientist",
"representation", "boils" — real words. the cache creates a **fixed attractor**
that overrides input. the mechanism works but converges too aggressively to one
point instead of being input-sensitive.

**plasticity test**: can't work while cache collapses generation. compact_memory
fires (novelty ~0.87, sync_delta ~41) but output doesn't change because the
cache attractor dominates. even 5x reinforced compaction had no effect.

### lesson learned (the hard way)

**cache-aware training should have been on from step 0 at K=1.** we trained 9k
steps pure (no cache exposure), then tried to bolt it on. the model treats cache
as noise because it never needed it during formative training. the ramp helps but
1000 steps of gradual exposure can't undo 9000 steps of cache-ignorant learning.

for the next training run: `--cache-aware-ratio 0.30` from step 0, even at K=1.

### plan: keep training

not changing architecture. the attractor pattern (real words, just stuck in a
loop) means the mechanism is functioning — it just needs more training time at
high cache-aware ratio to learn input-sensitive cache usage instead of collapsing
to a fixed point.

switching from H200 to cheaper H100 80GB (batch=1, same K=2). let it grind at
0.30 for thousands of steps without interruption.

### checkpoint inventory (all backed up locally)

| checkpoint | bpb | cache-aware | files |
|---|---|---|---|
| ffn_d12_clean 10k | 0.938 | n/a | model+optim+meta |
| ctm_d12_v2 9k | ~1.0 | 0% | model+optim+meta |
| ctm_d12_v2 9.5k ramped | ~0.98 | 0.15 | model+optim+meta |
| ctm_d12_v2 10k ramped | ~0.98 | 0.25 | model+optim+meta |

## step 10200 — ramp at 0.30, tick analysis (2026-03-11)

### ramp reached 0.30

cache-aware ratio hit target 0.30 at step ~10000. loss delta was **negative** (-0.11)
on the final bump — model adapting well. loss bouncing 3.2-3.5, no clear downward
trend yet over the 1200 steps since ramp started.

### tick diagnostic flip — concerning

early (step ~9000): tick1=4.0 tick2=3.3, tick2 better, selected 40/60%
latest (step ~10200): tick1=3.5 tick2=4.5, **tick1 now better**, selected 59/41%

the model FLIPPED — tick 2 went from being the refining step to being worse than
tick 1. interpretation: cache-aware training introduces so much disruption through
cross-attention that the model burns tick 1 just absorbing/filtering cache noise.
tick 2 has nothing useful left to do.

analogy: trying to read while someone's shouting. all your energy goes to filtering
noise, none left for comprehension. K=2 doesn't have enough headroom — tick 1
handles cache, tick 2 is wasted.

### decisions

- **don't bump K yet** — if the model can't use tick 2 productively, tick 3 would
  also be wasted. need to get K=2 working first.
- **don't start over** — 10k steps of language knowledge is in the weights.
- **let it cook** — run to 15k at K=2, cache-aware 0.30. give the model time to
  learn cache handling without changing anything.
- **design review needed** — use training time to review CTM architecture and
  identify what could be improved for the next run.

### recipe for next training run (learned the hard way)

1. cache-aware from step 0 (not bolted on at step 9000)
2. start at K=3+ so there's headroom for cache absorption + reasoning
3. warm-start from FFN checkpoint with cache-aware training from the beginning
4. don't change K, batch, or architecture mid-training

## step 11000→11500 — K=3 bump, generation collapse (2026-03-11)

bumped K=2→3 at step 11000. K=3 batch=3 OOMed (139GB), dropped to batch=1 (63GB).
loss spiked 3.4→3.98, recovered to 3.3 by step 11500. tick selection: 42/32/26%.
new tick 2 started at 47% (exploring), settled to 26% (not earning its keep).

### the real finding: CTM generation is broken

**CTM ON** (ticks running): garbled output.
"The capital of France is Paris. (A 135' FVA. The. in in 5 was..."

**CTM OFF** (bypass to FFN path): coherent output.
"The capital of France is Paris, in which there are three capital days..."

the language backbone works fine. the CTM tick loop corrupts autoregressive generation.
this is true at both 11000 (K=2) and 11500 (K=3). all prior "good" generation tests
were actually running with `use_ctm=False` — we were testing the FFN path unknowingly.

### the plasticity catch-22

- compact_memory writes into CTM synapse weights
- with CTM OFF: generation is coherent but compact_memory changes are invisible
- with CTM ON: compact_memory shifts distribution but output is garbled
- plasticity can't work until CTM generation works

### statistical plasticity test (step 11000, N=20)

ran comprehensive test: 30x compact + 50-step consolidation + combo.
- "sky is purple": 0/20 purple, but blue went 4→7/20 (domain shift)
- "name is Tommi": 0/20, but Tom-variants appeared ("tom stich", "thomas")
- "cats have wings": 0/20 across all methods

compact_memory nudges the distribution but can't inject specific tokens.

### why CTM generation fails

hypothesis: training uses teacher forcing (all tokens see ground truth context).
at inference, each token depends on previous CTM output → errors compound through
the tick loop. the sync accumulators amplify errors across tokens. this is a
training/inference mismatch specific to CTM's iterative computation.

### K=6 next

63GB at K=3 batch=1, 140GB available. bumping to K=6 to explore deeper thinking.
save-every=100 for fine-grained tracking. the model may plateau but we need data
on how higher K behaves.

### revised understanding

the critical path is not more K or more training. it's making CTM generate coherently.
without that, plasticity is stuck in a catch-22. the FFN backbone carries the language
knowledge; CTM needs to learn to use it without corrupting the signal.

## the breakthrough: single CTM architecture (2026-03-11)

### what we discovered

read the actual CTM paper. they use **one CTM with 50-100 ticks** on top of a backbone.
we had **12 CTMs with 6 ticks each**, one replacing every MLP layer. this is wrong:

- 12 independent CTMs can't build coherent thought — no shared state
- each CTM gets only K ticks, not enough for dynamics to converge (paper uses 50+)
- multi-tick loss only trained last layer — other 11 CTMs were dead weight
- explains why only layer 11 was activating in earlier diagnostics
- explains why CTM generation was garbled — 11 untrained CTMs corrupting the signal

the paper never did language modeling. we're first. but their architecture is clear:
backbone processes input, one CTM reasons over the output.

### the fix: `--ctm-layers=last`

new config option: only layer 11 gets CTM, layers 0-10 stay FFN.
warm-start from FFN 10k — layers 0-10 keep trained MLP weights, layer 11 fresh CTM.

### immediate results (K=1, single CTM)

| metric | 12 CTMs (K=3) | 1 CTM (K=1) | improvement |
|--------|---------------|-------------|-------------|
| ms/step | 21,000 | 780 | **27x faster** |
| tok/sec | 2,900 | 79,000 | **27x throughput** |
| MFU | 1.2% | 11.7% | **10x efficiency** |
| params | 892M | 521M | 42% smaller |
| loss at start | 3.98 (spike) | 3.28 | better baseline |

100k steps = 6.1B tokens in ~22 hours. previous architecture would take weeks.

### the plan

1. K=1 single CTM, cache-aware 0.30, 100k steps (~22 hours, 6B tokens)
2. ramp K when plateau — each K bump adds thinking depth to one coherent CTM
3. test generation with CTM ON — should work now since only 1 CTM, not 12
4. test plasticity — compact_memory writes to one place, no fragmentation
5. all the machinery (sync, trace, convergence) works as designed for first time

## single CTM: step 1500-2200, generation works, plasticity weak (2026-03-11)

### generation with CTMCache — IT WORKS

first ever coherent generation with CTM ON and CTMCache active:

```
Prompt: "The meaning of life is"
Output: "The meaning of life is well known in many respects. It was once one
of many reasons for the exploitation of life in India, China and other nations."

Prompt: "Once upon a time there was a"
Output: "Once upon a time there was a little bird in a tunnel - it was at one
of those early days - more scared away than they looked at it."
```

cache-aware training at 0.30 ratio is working. model generates coherently through
CTMCache — the state accumulation doesn't corrupt output. this was the #1 blocker
from the old 12-CTM architecture.

### auto K-ramp enabled

plateau detection: eval every 100 steps, window=300, threshold=0.005 bpb,
patience=2 strikes. when val bpb stalls for 600+ steps, K auto-bumps.

val bpb trajectory:
```
step    bpb
1500    1.039
1600    1.024  (best so far)
1700    1.026
1800    1.027
1900    1.029  (strike 1/2)
2000    1.030  (would have been strike 2, but crashed — see below)
2100    1.026  (improved, strikes reset)
```

### K-ramp crash: Muon optimizer shape mismatch

at step 2000, plateau detector triggered K 1→2. crash:
```
RuntimeError: output with shape [1, 1, 768] doesn't match the broadcast shape [1, 2, 768]
```

root cause: `param.data = new_embed` changes tensor data but Muon's momentum
buffers reference the old shape. fix: replace the entire `nn.Parameter` object,
not just `.data`:
```python
block.mlp.tick_embed = nn.Parameter(new_embed)  # new object, clean slate
```

### checkpoint_manager bug: mixed MLP/CTM loading

`build_model()` checked `has_mlp_checkpoint = any('.mlp.c_fc.' in k ...)` globally.
with `ctm_layers=last`, layers 0-10 still have MLP → always True → incorrectly
stripped layer 11's CTM weights on every load.

fix: check if the specific CTM layers have MLP keys (`c_fc`), not globally.
layer 11 has CTM keys (not `c_fc`), so it's left alone.

### plasticity test: mechanism works, signal too weak at K=1

tested compact_memory with probability probing:

```
fact: "My name is Tommi. I live in Helsinki."
probe: "Tommi lives in" → P(Helsinki)

BEFORE compact: 0.000047
AFTER compact:  0.000047  (zero change)
```

weights DO change (verified): c_proj delta norm = 0.0007, but total weight
norm = 21.6. that's a 0.003% update — invisible to the output.

why: at K=1, only 1 tick runs. sync statistics accumulate nothing meaningful.
the paper uses T=50-100 ticks for good reason — convergence needs iterations.
compact_memory is writing real updates, but the sync signal from 1 tick is noise.

**plasticity needs K >> 1.** waiting for auto-ramp to increase K before retesting.

### wandb restored

logged in as hitchhooker, project: nanoctm. live monitoring at:
https://wandb.ai/hitchhooker/nanoctm
