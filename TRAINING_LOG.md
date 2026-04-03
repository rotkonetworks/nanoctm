# training log

a record of our model's journey from noise to something that thinks.

## contents

- [step 0 — launch](#step-0--launch-2026-03-07)
- [step 500 — first checkpoint](#step-500--first-checkpoint-2026-03-08)
- [step 1000 — it knows words](#step-1000--it-knows-words-2026-03-08)
- [migration — 4x H100 attempt](#migration--4x-h100-attempt-2026-03-09)
- [step 11000 — plasticity fails on qwen3, c_proj rank bottleneck](#step-11000--plasticity-fails-on-qwen3-c_proj-rank-bottleneck-2026-03-13)
- [step 12000 — plasticity still fails, two-CTM architecture](#step-12000--plasticity-still-fails-two-ctm-architecture-2026-03-13)
- [Qwen3.5-4B — failed experiment, VRAM killed multi-tick](#qwen35-4b--failed-experiment-vram-killed-multi-tick-2026-03-15)
- [conclusions — what we know and the path forward](#conclusions--what-we-know-and-the-path-forward-2026-03-15)
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
- [migration to RTX 5090 BKK](#migration-to-rtx-5090-bkk--2026-03-11)
- [K-ramp 1→2 breaks generation](#k-ramp-12-breaks-generation-critical-finding)
- [the plan: how to succeed](#the-plan-how-to-succeed)
- [step 9000→9500 — cache-aware, flat 0.3](#step-90009500--cache-aware-training-flat-03-2026-03-10)
- [step 9000 (take 2) — adaptive ramp](#step-9000-take-2--adaptive-cache-aware-ramp-2026-03-10)
- [ramp tuning and restart saga](#ramp-tuning-and-restart-saga-2026-03-11)
- [step 10000 — plasticity test](#step-10000--cache-aware-ramp-complete-plasticity-test-2026-03-11)
- [step 10200 — tick analysis](#step-10200--ramp-at-030-tick-analysis-2026-03-11)
- [step 11000→11500 — K=3 bump, generation collapse](#step-1100011500--k3-bump-generation-collapse-2026-03-11)
- [the breakthrough: single CTM architecture](#the-breakthrough-single-ctm-architecture-2026-03-11)
- [single CTM: step 1500-2200, generation works, plasticity weak](#single-ctm-step-1500-2200-generation-works-plasticity-weak-2026-03-11)
- [K=18 warm tick experiment](#k18-warm-tick-experiment--step-3400-3412-2026-03-11)
- [why K-ramp is fundamentally broken](#why-k-ramp-is-fundamentally-broken--code-level-analysis-2026-03-11)
- [pivot to Qwen backbone](#pivot-to-qwen-backbone--cutting-the-corner-2026-03-11)
- [three-factor neuroplasticity — compact_memory()](#three-factor-neuroplasticity--compact_memory-2026-03-12)
- [Qwen3-0.6B + CTM K=32 full-featured training](#qwen3-06b--ctm-k32-full-featured-training-2026-03-12)
- [ctm.rotko.net — live site and debugger](#ctmrotkonet--live-site-and-debugger-2026-03-12)

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

---

## migration to RTX 5090 BKK — 2026-03-11

H200 nuked before full checkpoint transfer. saved CTM step 2500 model weights (1.3GB,
no optimizer) to local machine. uploaded to new RTX 5090 in Bangkok (10ms ping, 12MB/s
upload — domestic Thai network vs 8KB/s to the old US server).

resumed CTM training from step 2500 without optimizer state. optimizer momentum rebuilds
in ~200 steps. torch.compile works on this machine (CUDA 12.8, SM 120). ~2.7s/step at
K=1 with compile, ~3.8s/step at K=2.

### K-ramp 1→2 breaks generation (critical finding)

**this is the most important finding of the session.**

plateau detection triggered K-ramp from 1→2 around step 3400. val bpb continued
improving after the ramp (1.009 → 0.980). training looked healthy. but generation
is completely broken:

```
step 3000 K=1: "life is so simple and fast that it can be completed without..."  ← coherent
step 3400 K=1: "life in life Earth or space. It is also called life..."          ← coherent
step 3500 K=2: ",, can can be viewed as a difficult one. The is of the..."       ← garbage
step 4000 K=2: "well known. Can the one and is is the the more..."               ← garbage
```

val bpb at step 4000 is 0.980 — the best we've ever seen. but the model can't generate
a single coherent sentence. **bpb improvement does not equal generation quality.**

### diagnosis: why K-ramp breaks generation

the K-ramp introduces a brand new tick (tick 1) with random 0.01-scale embeddings.
during training, the model learns to use both ticks — tick selection at step 4000 is
roughly 46%/54%, so both ticks matter. but:

1. **tick 1 starts random.** its contribution through cross-attention, synapse, and
   NLM is essentially noise. the model is learning to route through this noise.

2. **val bpb measures teacher-forced next-token prediction.** the model sees the correct
   previous tokens and predicts the next one. it can get good bpb by learning to use
   tick 0 (the pretrained one) for most tokens and tick 1 for easy completions. but in
   autoregressive generation, errors compound — tick 1's noisy contributions cascade.

3. **scheduled sampling should fix this** — we have it at 10%. but 10% may not be enough.
   the model needs to practice generating with its own mistakes at K=2, not just see
   teacher-forced sequences.

4. **optimizer cold-start may contribute.** we resumed without optimizer state, so
   momentum for all parameters was zero. the first ~200 steps after resume were
   effectively random walk. K-ramp happened at step 3400 (900 steps after resume),
   so optimizer should have been warm by then. but the combination of cold optimizer
   + K-ramp within the same run is suspect.

### also found: checkpoint meta doesn't save current K

`model_config_kwargs` was set once at init and never updated after K-ramp. every
checkpoint after K-ramp saved `ctm_iterations=1` in meta even though the model was
running K=2. the checkpoint loader detects the mismatch from tick_embed shape and
fixes it, but this is fragile. fixed: `ramp_k()` now updates `model_config_kwargs`.

### plasticity at K=2: mechanism works, output still garbage

compact_memory() runs correctly — 91.9% of params changed, novelty gating at 50%,
c_proj weight change 0.62% at lr=1e-4. but since the model can't generate coherent
text at K=2, recall tests are meaningless. **plasticity can't be tested until
generation works.**

aggressive compaction (lr=1e-3, 5 rounds) destroyed the model further — c_proj
changed 75.6%, outputs became dots and commas. clearly too aggressive for a model
that's already struggling.

## the plan: how to succeed

### option A: fix K-ramp (best path)

the K=1 model generates well. the problem is specifically the K-ramp transition.

1. **increase scheduled sampling.** at K-ramp time, temporarily boost scheduled
   sampling from 10% to 50%. the model needs heavy exposure to its own K=2 outputs
   during training. after generation stabilizes (test every 100 steps), reduce back
   to 10-20%.

2. **gradual tick mixing.** instead of adding tick 1 at full strength, scale its
   contribution from 0→1 over 500 steps. the model slowly learns to incorporate
   the new tick without disruption. this requires modifying CTMBlock to support
   a tick_weight schedule.

3. **warm-start tick 1.** instead of random init for the new tick_embed, initialize
   it as a perturbation of tick 0: `tick_1 = tick_0 + 0.01 * randn`. the new tick
   starts as a near-copy of the working tick, then differentiates through training.

4. **eval with generation, not just bpb.** add a generation quality check every N
   steps (generate a few sentences, compute perplexity of the generated text using
   the model itself, or just log samples). bpb alone is misleading.

### option B: stay at K=1, test plasticity differently

if K-ramp keeps breaking generation, stay at K=1 and try:

- **online learning** (Session with online_lr > 0): gradient-based learning from
  conversation, doesn't need sync statistics at all. this is phase 4 in our roadmap.
- **chunked BPTT** first (phase 2): teach the model what persistent state means
  before trying to accumulate sync patterns.

### option C: skip K-ramp entirely, train at target K from start

train a fresh model at K=5 or K=8 from the beginning (with warm-start from FFN).
no ramp, no transition — the model learns all ticks together. slower training but
avoids the ramp problem entirely.

### recommended path

**option A, approach 3 (warm-start tick)** — resume from step 3400 (K=1, good
generation), ramp to K=2 with tick_1 initialized from tick_0 + noise, high
scheduled sampling. test generation after 100 steps. if coherent, continue.
if not, fall back to option C.

## K=18 warm tick experiment — step 3400-3412 (2026-03-11)

tried plan A3 aggressively: jumped straight from K=1 to K=18 with warm tick init
(all ticks initialized as tick_0 + 0.01*randn). 50% scheduled sampling. BKK 5090.

### results after 12 steps

- **loss dropping**: 3.63 → 3.33 (good sign, model is learning)
- **tick differentiation happening**: ticks 0-11 loss ~3.58, ticks 16-17 loss 5.2/7.15
- **certainty diverging**: ticks 0-11 cert ~0.72, tick 17 cert 0.24
- **dream converged**: K 18→2 (only 2 ticks useful for convergence)
- **generation at step 3400: GARBAGE** — "capital of France is the Vriavhever"

### diagnosis

the warm tick init prevents immediate training collapse (loss does drop), but
generation is still broken. the backbone has only 3400 steps of FFN training —
it doesn't know language well enough to support 18 thinking iterations. the CTM
layer is trying to learn language modeling AND tick differentiation simultaneously.

the later ticks (16-17) are actively diverging — their loss is INCREASING while
early ticks improve. this suggests the model is learning to ignore later ticks
(certainty 0.24) but they still contribute noise during autoregressive generation.

### revised plan: strong FFN backbone first

the right approach is probably:

1. **train FFN for 50-100k steps** — solid backbone that knows language (bpb ~0.85)
2. **replace layer 11 FFN with CTM at high K** — warm-start from FFN weights
3. the CTM only needs to learn "how to think" not "what words are"

economics on 5090 (pure FFN, no CTM overhead):
- ~500-800ms/step, 100k steps ≈ 14-22 hours
- FFN bpb 0.938 at 5k steps, probably ~0.85 at 100k

economics on H100 80GB (when we get one):
- K=32+ easily fits in memory
- ~200ms/step FFN, 100k = 5.5 hours
- then CTM at K=32 with strong backbone

**key insight**: the foundation must be solid before you build the thinking layer.
we were trying to teach a toddler calculus — need to teach it language first.

## why K-ramp is fundamentally broken — code-level analysis (2026-03-11)

after three separate attempts at K-ramp (K=1→2, K=1→5, K=1→18), all producing
garbage generation despite improving bpb, we traced the root cause to the sync
accumulator math in `CTMBlock.forward()`.

### the mechanism

the CTM output is NOT selected from one tick. it's an exponentially-weighted
accumulation across ALL ticks:

```python
# each tick k updates the sync accumulators (lines 572-584):
alpha_out = r_out * alpha_out + pp_out   # exponential moving average
beta_out  = r_out * beta_out  + dopamine

# final output after all K ticks (line 597):
synch = alpha_out / sqrt(beta_out)
out = c_proj(synch)
```

`alpha_out` after K=1 tick is a COMPLETELY DIFFERENT VALUE than after K=18 ticks,
even with identical tick embeddings. the decay `r_out` down-weights earlier ticks,
later ticks dominate. `c_proj` was trained to map K=1's sync distribution to good
logits. K=18's sync distribution is a different animal.

this is why:
- **bpb improves**: the multi-tick loss (argmin over ticks) cherry-picks the best
  tick per token during teacher-forced training. more ticks = more chances to get
  lucky. but the ACTUAL output uses the accumulated sync, not the best tick.
- **generation collapses**: during autoregressive generation, the model uses its
  own outputs. the shifted sync distribution produces slightly wrong logits →
  slightly wrong tokens → next step's sync is even more wrong → cascade.

### it's not a bug, it's the math

the sync accumulator is a sum, not a mean. adding more terms changes the output.
there's no code fix that would make K-ramp "just work" without one of:

1. **K-normalization**: divide sync by K so output is mean, not sum. untested,
   might kill the model's ability to use different amounts of thinking per token.
2. **train at target K from step 0**: no transition, no distribution shift. this
   is the correct path.
3. **freeze-and-thaw**: after K-ramp, freeze everything except tick_embed and
   c_proj for N steps so the output projection relearns the new sync distribution.
   untested but theoretically sound.

### mistakes we made (honest accounting)

1. **tried K-ramp 3 times** before understanding why it fails. should have read
   the sync accumulator math after the first failure instead of blaming tick init.

2. **trusted bpb as quality signal**. bpb improved with higher K because multi-tick
   loss is an optimistic metric (argmin over ticks). generation quality is the real
   test, and we didn't check it early enough.

3. **warm tick init was a red herring**. it helps training stability (loss drops
   faster) but doesn't solve the fundamental sync distribution shift. we spent time
   implementing and debugging it when the real issue was architectural.

4. **tried to ramp from weak backbone**. with only 3400 FFN steps, the backbone
   barely knows language. mounting CTM on it means the CTM has to learn language
   AND thinking simultaneously — too much for one module.

5. **didn't study the paper's training protocol carefully enough**. the CTM paper
   trains at fixed K throughout. K-ramp was our invention, and it doesn't work
   because the sync accumulator isn't K-invariant.

### the real path forward

- train FFN backbone for 50-100k steps (bpb ~0.85)
- mount CTM at target K from the first CTM step (K=18 on 5090, K=32 on H100)
- never change K during training
- always validate with generation quality, not just bpb

## pivot to Qwen backbone — cutting the corner (2026-03-11)

### the problem we're trying to solve

training our own FFN backbone to bpb ~0.85 takes 100k steps (~13 hours on 5090).
that's 13 hours before we can even *start* working on plasticity — the actual goal.
and our d12 backbone (768-dim, 12 layers, 469M params) trained on climbmix is
a toy compared to what's out there. we'd be mounting CTM on a mediocre language model.

### the insight

Qwen2.5-0.5B exists. 24 layers, 896-dim, 14 attention heads, 151k vocab. trained on
18 trillion tokens by a professional lab. it already knows language better than our
d12 FFN ever will at 100k steps. why train a backbone when someone already made a
great one?

the CTM paper mounts one CTM on top of a backbone anyway. our single-CTM architecture
(the breakthrough from earlier today) already proved that only the last layer needs
to be CTM. so: freeze Qwen, replace its last MLP with our CTMBlock, train only the
CTM parameters.

### what we built

`nanochat/qwen_backbone.py` — QwenBackboneGPT wrapper:

- loads Qwen2.5-0.5B via HuggingFace, freezes all 481M backbone params
- replaces layer 23's MLP with our CTMBlock (52M trainable params)
- manages its own HF DynamicCache for KV (our KVCache is Flash Attention style)
- forward pass iterates all 24 layers, intercepting layer 23 for CTM
- compatible with Engine for generation, dream() for diagnostics
- multi-tick loss, CTM-only checkpointing

QwenTokenizer wraps HF's tokenizer with our nanochat interface.

### test results on RTX 5090

```
Total params: 533,080,944
Trainable (CTM): 52,122,608 (9.8%)
Frozen (backbone): 480,958,336 (90.2%)

Forward pass: OK (loss 5.22 — untrained CTM, expected)
Generation: coherent Qwen output (CTM is no-op with zero-init c_proj)
CTM convergence: layer 23 converges in 4 ticks (delta=0.002)
Gradient flow: c_proj gets grads, backbone frozen
```

generation samples with untrained CTM (backbone doing all the work):
- "The capital of France is" → "A. Paris B. Lyon C. Bordeaux D. Marseille"
- "Once upon a time" → "there was a old... but now there is a new..."
- "The meaning of life is" → "the question that humanity has been pondering..."

### what led us here — the K-ramp saga

the road to Qwen was paved with hard lessons. we spent days believing K (thinking
iterations) was adaptable — that we could start at K=1 and ramp up as the model
learned. every K change broke generation. every time. the sync accumulator normalizes
by sqrt(beta), so output distribution shifts when K changes. we tried K=1→2, K=2→3,
K=1→3→5, K=1→18 with warm ticks. all failed the same way: bpb looked fine (argmin
over ticks is optimistic) but generation collapsed.

the remarkable thing: the model kept recovering. given enough compute after each
K-change, it would claw back to coherent generation. but this was on the 12-layer
all-CTM architecture — our other big mistake. 12 CTM layers meant 11 untrained CTMs
corrupting signal before layer 11 could think. 27x slower (21s vs 780ms/step).
we were burning compute recovering from self-inflicted wounds on an architecture
that was fundamentally wrong.

once we discovered single-CTM (one CTMBlock on the last layer, matching the paper),
everything got faster but K-ramp still broke. that's when we finally read the math
and understood: K must be fixed from the start of CTM training, period. which means
you need a strong backbone first. which means either train FFN for 13 hours... or
just use Qwen.

### why this is the right move

1. **skip 13 hours of FFN training** — backbone is already world-class
2. **10x more data behind the backbone** — 18T tokens vs our ~2B token budget
3. **focus on what matters** — plasticity is the goal, not language modeling
4. **still our architecture** — CTMBlock is unchanged, same sync, same synapses
5. **can always go back** — FFN d12 checkpoint at step 10k saved locally

### what we're giving up

- full control over the backbone (can't unfreeze without OOM)
- our custom tokenizer (switched to Qwen's 151k BPE)
- reproducibility on tiny GPUs (need ~3GB VRAM minimum)
- the satisfaction of training from scratch

### risks

- CTMBlock was designed for 768-dim, now running at 896-dim. should be fine
  (it's parameterized) but untested at scale
- HF DynamicCache vs our KVCache — different code paths for generation
- Qwen's attention uses GQA (14 Q heads, 2 KV heads) — our CTM doesn't
  touch attention so this shouldn't matter

### FFN d12 backup

FFN 100k training ran to step ~10.9k before we stopped it. checkpoint at step 10k
saved locally. can resume anytime if Qwen approach doesn't pan out.

next: write training script integration and start CTM training on the 5090.

---

## three-factor neuroplasticity — compact_memory() (2026-03-12)

### the question

can we teach the model a fact, consolidate it into weights, and verify recall
from a fresh context? this is the whole point of the project. not language
modeling — *memory*.

### attempt 1: pure gradient descent (cheating)

first version of `compact_memory()` was just Adam + loss.backward() on the
teaching text. 50 steps, lr=1e-3. it worked — loss dropped from 2.4 to 0.3,
model could recite "Tommi" and "Helsinki". but it was pure overfitting on 77
tokens. degenerate looping: "nameHelloI am nameHelloI am". this isn't
plasticity, it's just fine-tuning.

### attempt 2: pure Hebbian (too weak)

tried a pure Hebbian update: ΔW = η × pre × post, using the sync accumulators
as co-activation traces. only touched start_state (delta 0.026), start_trace
(delta 6.29), decay (delta 0.05). no recall at all. the problem: pure Hebbian
can't solve credit assignment through the SynapseUNET's deep layers. local
learning rules can't reach non-local weight matrices.

### the literature

searched for anyone who had combined Hebbian learning with gradient descent.
found a surprising amount of theoretical grounding:

**1. three-factor learning rules (Gerstner et al., 2018)**
ΔW = η × pre × post × M, where M is a neuromodulatory signal (dopamine,
noradrenaline, etc). the third factor solves the credit assignment problem that
pure Hebbian can't — it gates which synaptic changes get consolidated based on
a global reward/error signal.

**2. weight decay IS Hebbian homeostasis (arXiv 2505.18069)**
"How Weight Decay Learns Local Hebbian Plasticity Rules" — regularized gradient
descent can be decomposed into activity-dependent synaptic updates that are
locally Hebbian. this means AdamW with weight decay is *already* doing something
Hebbian, not cheating. the gradient provides global error signal, weight decay
provides local homeostatic regulation.

**3. DA-SSDP: dopamine-gated synchrony plasticity (arXiv 2512.07194)**
dopamine-modulated spike synchrony-dependent plasticity. the gate:
`G_b = clip(1 + k(S_b - μ_S)/σ_S, 0, 2)` modulates synaptic updates based on
synchrony deviation from population mean. directly maps to our sync
accumulators — S_out and S_action ARE synchrony measures.

**4. heterogeneous RPEs (Cell Reports 2025)**
dopamine doesn't encode a single reward prediction error. different DA pathways
carry pathway-specific prediction errors. maps to our per-token dopamine: each
token gets its own surprise signal, not a single scalar for the whole batch.

**5. Backpropamine (Miconi et al., ICLR 2019)**
a Hebbian trace is maintained alongside regular weights, modulated by a
neuromodulatory signal, and the whole system is trained end-to-end with
backprop. our approach is similar but uses the sync accumulators as the
Hebbian trace rather than a separate matrix.

### the solution: three-factor wake→encode→sleep

the key insight: gradient descent and Hebbian learning aren't competing
approaches. they're complementary. gradient descent solves credit assignment
(reaching deep weights), Hebbian traces solve *what to consolidate* (which
synapses were active during important events), and dopamine gates *how much*
to consolidate (prediction error = surprise = importance).

implemented as a three-phase consolidation cycle in `compact_memory()`:

**wake phase** — compute dopamine from prediction error:
- forward pass the teaching text through frozen model
- per-token cross-entropy = prediction error
- normalize relative to mean, clamp to [0.5, 2.0]
- high CE (surprising) → high dopamine → remember this
- low CE (predictable) → low dopamine → don't bother

**encoding phase** — dopamine-gated sync trace:
- re-forward with per-token dopamine tensor in CTMCache
- sync accumulators now weight each token by its surprise
- the cache becomes a dopamine-shaped memory trace
- surprising tokens dominate sync patterns
- this IS the eligibility trace from three-factor rules

**sleep phase** — replay with sync-modulated gradients:
- gradient descent on teaching text (the "replay")
- sync-modulated weight decay: active pairs get lower decay (protect memory)
- per-parameter learning rate groups (start_state/trace get 2x, tick_embed 0.3x)
- gradients scaled by dopamine-weighted sync importance
- three-factor equation: gradient × sync_importance ≈ pre × post × modulator

### per-token dopamine in CTMBlock

modified `CTMBlock.forward()` to support (BT,) dopamine tensors, not just
scalar. the sync accumulation code already multiplied by dopamine — tensor
support "just works" with broadcasting:

```python
dopamine = ctm_cache.dopamine if ctm_cache is not None else 1.0
if isinstance(dopamine, torch.Tensor) and dopamine.dim() >= 1:
    dopamine = dopamine.view(BT, 1)  # broadcasts with (BT, n_synch)
```

### test results

model: Qwen2.5-0.5B backbone + CTM K=32, 1000 training steps.
teaching: 5 sentences about "Tommi from Helsinki", 77 tokens.
compact_memory: lr=3e-4, 30 replay steps.

**wake phase:**
```
Dopamine: mean=0.918, min=0.500, max=2.000, CE_mean=2.422
```

**encoding phase:**
```
dS_out=41.24 (vs 58.24 without dopamine — dampened predictable tokens)
dS_act=similar dampening
```

**sleep phase:**
```
Losses: [2.42, 2.18, 1.98, 1.81, ... 1.08]  (smooth convergence, no collapse)
Total delta: meaningful parameter changes across all CTM components
```

**recall from fresh context (no prior teaching in context):**
```
"My name is" → "Tom. I am from my mum."
"What is my name?" → "Your name is your name, remember"
"My name is Tommi and I" → "am from Helsinki. I am from Helsinki, Helsinki is my home"
"What do you know about me?" → "I am a born-Remembering"
```

the model recalls fragments of the teaching — "Tom", "Helsinki", "remember",
"your name is" — from a completely fresh context. it's noisy and repetitive,
but the information transferred from a 77-token experience into persistent
weight changes. with only 1000 base training steps, this is promising.

### what this means

1. **compact_memory() works** — information survives context reset
2. **dopamine gating shapes what gets remembered** — surprising tokens get
   higher sync weight, boring tokens get dampened
3. **the architecture supports it** — sync accumulators are natural eligibility
   traces, dopamine gating fits without architectural changes
4. **it's not pure fine-tuning** — the three-phase cycle with dopamine gating
   and sync-modulated weight decay produces qualitatively different results
   than raw gradient descent (less degenerate looping, more coherent recall)

### next steps

- more base training (1000 steps is barely anything)
- multi-fact teaching (can it learn multiple facts without interference?)
- forgetting curves (does recall degrade over continued training?)
- conversational teaching (dialog format, not raw text repetition)

## Qwen3-0.6B + CTM K=32 full-featured training (2026-03-12)

### why the restart

the previous Qwen2.5-0.5B run (5k steps, `qwen25_ctm_k32_cont`) was wasted:
`--cache-aware-ratio` defaulted to 0.0. five thousand steps of training and the
model never once saw its own CTMCache. every feature we built for memory —
episodic recall, cache warm-starting, sleep consolidation — was dead code.

also discovered that multi-tick certainty loss OOMed on Qwen's 151k vocab.
the CTM paper (arXiv 2505.05522) computes loss at ALL K ticks with full grad —
fine for CIFAR (V=10), catastrophic for language (V=151,936). each tick's logits
are [4096, 151936] = 2.4GB. with K=32 in the computation graph: ~77GB.

### architecture

- **backbone**: Qwen3-0.6B (28 layers, D=1024, 742M frozen params)
- **CTM**: single block at layer 27 (67.8M trainable params)
- **total**: 810M params, 8.4% trainable
- **K=32** thinking iterations per token

### what's enabled (all from step 0)

1. **cache-aware training** (`--cache-aware-ratio 0.30`): 30% of steps split
   sequence in half, forward first half to build CTMCache (no grad), train on
   second half WITH cache. the model learns to use its own memory.

2. **multi-tick certainty loss** (VRAM-efficient): per-token tick selection,
   loss = best-tick-loss. early ticks (0 to K-5) detached to limit VRAM —
   truncated BPTT through last 4 ticks only. diagnostics computed for all 32.
   loss proxy for certainty (avoids V-sized softmax per tick).

3. **sleep consolidation** (every 50 steps): gradient-free Hebbian update on
   replay buffer. certainty × accuracy → weight scaling. no backward pass,
   zero VRAM overhead.

4. **SFT data mixing** (`--sft-ratio 0.05`): 5% of steps use custom SFT data
   (820 texts: identity, memory conversations, values, blog posts, neuroscience).

5. **per-tick wandb diagnostics** (every step): loss and selection percentage
   for all 32 ticks logged to wandb. certainty computed for final tick only.

6. **replay buffer** (16 batches on CPU): stores recent (x, y) pairs for
   consolidation sampling.

### the multi-tick OOM saga

three crashes before it ran:

1. **OOM in `_multi_tick_loss`**: softmax over V=151k at each of 32 ticks.
   fix: two-pass approach — pass 1 (no_grad) finds best ticks, pass 2 (with
   grad) only recomputes selected ticks.

2. **OOM in CTM forward**: even with two-pass loss, storing 32 `tick_outputs`
   in the computation graph retained activation memory for all ticks.
   fix: detach early tick outputs (`tick_out.detach()` for k < K-4). only
   last 4 ticks retain gradient — truncated BPTT.

3. **"element 0 does not require grad"**: when ALL tokens selected detached
   ticks, the loss had no grad_fn. fix: always blend in last-tick loss as
   anchor. if tokens prefer early ticks, last-tick loss provides gradient;
   the blend weight tracks what fraction of tokens chose grad-enabled ticks.

### early results (step 0-296)

```
step    loss    note
0       2.94    fresh CTM on frozen Qwen3 backbone
50      3.26    first consolidation (loss=3.54, cert=85%)
100     3.23    dream: L27 converged, delta=0.023
                ticks: t0:17.7%, t18:8.8%, t17:8.3% — bimodal
150     3.03    consolidation #3 (loss=2.91, cert=79%)
200     2.96    dream: L27 NOT converged, delta=0.112
                ticks: t0:13.9%, t3:9.4%, t2:7.2% — shifting early
296     2.91    loss steadily dropping, lr reaching 1e-3
```

speed: ~2.2s/step average (1.7s normal, 2.6s cache-aware). ~1,550 tok/sec.
VRAM: 24.6GB / 32.6GB (75%). ETA: ~33 hours to 50k steps.

### tick distribution — early observations

bimodal pattern emerging:
- **tick 0** dominates (14-18%): immediate/reflexive responses. the backbone
  already knows the answer, CTM just passes it through.
- **ticks 16-19** cluster (5-9%): deep thinking tokens. these are the ones
  where iterative refinement helps.
- **middle ticks** (5-15): low selection (~2-4%). model either knows immediately
  or needs many iterations — rarely benefits from moderate thinking.

this matches intuition: easy tokens → tick 0, hard tokens → late ticks.
as training progresses, expect the late-tick cluster to strengthen and possibly
shift later as the model learns to use more iterations productively.

### wandb

run: `qwen3_ctm_k32_full`
link: https://wandb.ai/hitchhooker/nanoctm

per-tick graphs should show:
- `ctm/tick_{k}/loss` for k=0..31
- `ctm/tick_{k}/selected_pct` for k=0..31
- `ctm/certainty/mean` and `ctm/certainty/std`
- `ctm/loss/argmin` and `ctm/loss/last_tick`
- `ctm/loss/grad_tick_frac` — how often tokens pick grad-enabled ticks

### the thesis

CTM + neuroplasticity may be the path to recursive self-improvement at small
scale. the architecture has three properties no frozen LLM has:

1. **iterative refinement**: K=32 ticks = 32x effective depth with shared
   weights. the model allocates compute adaptively per token.
2. **self-observation**: sync readout lets the model see its own processing
   state. tick selection = metacognition (knowing when you know).
3. **weight self-modification**: compact_memory() writes experience into
   weights. consolidation reinforces what works. the model rewires itself.

a 0.5B model that thinks for 32 steps and rewires from experience may
outperform a 16B model that does one pass and forgets everything.

next milestones:
- step 500: first checkpoint, test compact_memory() on Qwen3
- step 2000: test memory pipeline (teach → compact → restart → recall)
- step 5000: Claude-guided evaluation pass (generate → evaluate → compact)
- step 10000: compare to base Qwen3-0.6B on reasoning tasks

---

## ctm.rotko.net — live site and debugger (2026-03-12)

the project now has a public face: **https://ctm.rotko.net**

### what's live

three pages:

1. **index** — hero, architecture overview, and a live 3D CTM debugger built in
   Rust/egui/WASM. the debugger replays real training data from wandb — 2,250
   snapshots from the `qwen3_ctm_k32_full` run. you can watch tick weight
   distributions, loss curves, learning rate, and grad-tick fraction evolve in
   real time. **[see the debugger →](https://ctm.rotko.net/#debugger)**

2. **research** — technical writeup of the CTM architecture. covers:
   - why context window hacks (RAG, summarization, MemGPT) are not memory
   - backbone-agnostic CTM integration (any open-source transformer)
   - three-factor neuroplasticity (wake/encode/sleep)
   - recursive self-improvement: agents editing their own training code vs
     actual neuroplastic weight modification (with an honest AI safety warning)
   - CTM internals: SynapseUNET, SuperLinear NLMs, dual sync, CTMCache

3. **invest** — if people want to contribute or back the work, there's a way
   to reach out. not a pitch deck, just a door.

### infrastructure

- **stack**: nginx (static) + python stdlib (JMAP contact API), single container
- **mail**: contact form → Stalwart JMAP
- **deploy**: GitHub Actions → docker build → podman
- **routing**: HAProxy + Let's Encrypt SSL

### debugger data

the WASM debugger at [ctm.rotko.net/#debugger](https://ctm.rotko.net/#debugger)
loads `ticks.json` — 2,250 real training snapshots pulled from wandb. each
snapshot contains per-tick weight distributions across all 32 ticks: how the
model distributes its "thinking budget" across iterations.

overlays show:
- **bpb** (bits per byte) — the core loss metric
- **lr** (learning rate) — in scientific notation
- **grad%** — fraction of tokens selecting gradient-enabled ticks

this is live training data, not synthetic. as training progresses, we'll pull
fresh snapshots to show how the tick distribution evolves — early ticks for
reflexive answers, late ticks for deep thinking.

### step 1000 status

at the time of launch, Qwen3 + CTM K=32 is at step 1000:
- loss: 3.09, still early
- generation: not coherent yet ("uby Template:Usa The water boiling point...")
- convergence: delta=0.16, CTM hasn't found stable tick patterns
- GPU: 98% utilization, 25.8GB/32.6GB VRAM, 64°C on RTX 5090
- speed: 3.6s/step, 2,253 tok/sec
- ETA to 10k: ~9.3 hours

the site will evolve with the model. when plasticity works on Qwen3, the
research page gets updated. when we have a model worth talking to, we add
live inference. the site is the lab notebook made public.

---

## step 11000 — plasticity fails on qwen3, c_proj rank bottleneck (2026-03-13)

### the test

stopped training at step 11,000 (loss ~2.1, val bpb 4.31) to run a full test suite:

1. **generation quality**: good. backbone knowledge intact — fibonacci correct, security/crypto
   answers coherent. still textbook-ish (backbone talking, CTM barely steering).
2. **SFT absorption**: zero. "sync accumulators" returns SQL nonsense. "how does your CTM
   contribute" → collapse loops. at 5% ratio and 11k steps, model has barely seen SFT.
3. **session/multi-turn**: broken. hallucinations ("Ginny"), collapse. KVCache works but
   generation quality in conversation mode is rough.
4. **plasticity**: **0/4 recall.** compact_memory ran, loss dropped 8.82→5.23, but nothing
   stuck. "Rotko Networks is based in 2013" — hallucinating, not recalling.

even remapping KNOWN facts (France→Bangkok) fails — produces repetition collapse, not recall.
aggressive lr (1e-3) → "Networks Networks Networks Networks". the mechanism is broken.

### the diagnosis: c_proj is rank 3

deep analysis of CTM state revealed the smoking gun:

| metric | qwen3 @ 11k | qwen2.5 (where it worked) |
|--------|------------|---------------------------|
| c_proj rank at 90% energy | **3** | 61 |
| condition number | 55,351 | — |
| top singular value | 54.84 | — |
| 2nd singular value | 12.23 (4.5x smaller) | — |
| CTM contribution ratio | 0.89 | 0.76 |

the sync-to-residual projection collapsed into 3 effective dimensions. the CTM is active
(0.89 contribution ratio) but routes everything through 3 "superhighway" dimensions. when
compact_memory tries to write new facts, gradients flow along σ₁ (the dominant direction)
and overwrite existing signal → repetition collapse.

c_proj was zero-initialized and grew to rank 3 in 11k steps. training loss has no incentive
to use more dimensions — the model can minimize next-token loss with just 3 directions.
weight decay actively shrinks unused dimensions.

### why qwen2.5 worked and qwen3 doesn't

qwen2.5-0.5B: 24 layers, weaker backbone → CTM had to do more work across more dimensions.
qwen3-0.6B: 28 layers, stronger backbone → CTM can be lazy, squeezes into rank-3 corridor.
stronger backbone = narrower CTM contribution = worse plasticity.

the 3D tick debugger visualization confirms this — a few dominant spikes in an otherwise
flat sync landscape. the energy concentrates in a handful of neuron pairs.

### the fix: spectral regularization

added `--spectral-reg 0.1` to training. penalizes concentration of c_proj singular values:

```
concentration = σ_max² / ||W||_F²   (1.0 = rank-1, 1/min(m,n) = uniform)
loss += spectral_reg * concentration
```

uses power iteration (3 mat-vec products) to approximate σ_max — cheap, differentiable,
no full SVD needed. spectral metrics (rank90, rank99, concentration, condition number)
logged to wandb every 50 steps.

training restarted from step 11,000 with spectral reg. if rank90 doesn't expand past 5-6
after a few thousand steps, next option is unfreezing last 2 backbone layers.

### parallel: building our own observability

added POST /api/ingest to ctm.rotko.net server.py. training script POSTs snapshots
(tick data + spectral metrics) alongside wandb — our own telemetry pipeline feeding
the 3D debugger in real-time. replaced Python API with Rust (axum + tokio) — 1.5MB
static binary with WebSocket multicast for real-time streaming to frontend.

## step 12000 — plasticity still fails, two-CTM architecture (2026-03-13)

### the problem: single CTM can't penetrate frozen attention

spectral reg successfully expanded c_proj rank from 3 → 150 in ~1k steps. rank is no
longer the bottleneck. but plasticity STILL fails: 5 compact_memory configs tested at
step 12k, all returned 0/4 recall. CTM contribution ratio = 0.37-0.43 (vs 0.76 on
qwen2.5 where plasticity worked).

the core issue: one CTM at layer 27 can change the output (proved by mode collapse at
high lr) but can't TARGET specific tokens. it's like having a steering wheel that only
turns hard left or hard right — you can crash, but you can't navigate. the signal from
a single CTM block doesn't have enough leverage over 28 frozen attention layers.

### aggressive compact results (all configs, step 12000)

| config | lr | steps | recall | notes |
|--------|------|-------|--------|-------|
| standard | 3e-4 | 30 | 0/4 | loss drops, outputs garbage |
| gentle | 1e-4 | 50 | 0/4 | too weak to imprint |
| aggressive | 5e-4 | 40 | 0/4 | mode collapse |
| very aggressive | 1e-3 | 20 | 0/4 | catastrophic forgetting |
| c_proj only | 5e-4 | 60 | 0/4 | not enough params |

### the hypothesis: two-CTM like split brain

human cortex has two processing layers — the idea is CTM at layer 14 (mid-network)
injects signal that 13 frozen attention layers then amplify and route before the second
CTM at layer 27 makes the final adjustment. mid-layer CTM shapes the representation
space; top-layer CTM fine-tunes the output.

c_proj inits to zeros, so bolting a fresh CTM onto layer 14 is safe — zero output at
init means no signal corruption to the trained layer-27 CTM.

### implementation: multi-CTM QwenBackboneGPT

generalized the architecture to support N CTM layers via `--ctm-layers "14,27"`:
- `ctm_blocks` (ModuleDict) replaces single `ctm_block`
- forward loop intercepts all CTM layers; multi-tick loss on last CTM only
- checkpoint format: single-CTM saves flat dict (backward compat), multi-CTM saves keyed
- resume handles mixed: loads existing layer-27 weights, fresh-inits new layer-14
- all methods (dream, compact_memory, consolidate) operate across all blocks

### VRAM reality check

| config | batch | VRAM | speed |
|--------|-------|------|-------|
| 1-CTM K=32 | 2 | 16.1 GB | ~2.5s/step |
| 2-CTM K=32 | 2 | OOM (>31 GB) | — |
| 2-CTM K=32 | 1 (accum=2) | fits | ~5.5s/step |

two CTMs at batch=2 OOM on 5090 32GB — multi-tick loss pass (K × vocab recomputation)
is the bottleneck. batch=1 with grad_accum=2 fits. effective batch size same (4096 tokens),
but 2x slower per step.

training resumed from step 12000 with `--ctm-layers "14,27"` --device-batch-size 1.
135M trainable params (2 × ~52M CTM blocks + overhead). wandb: qwen3_ctm_2layer_k32.

key question: will the mid-layer CTM develop meaningful representations that the frozen
attention layers can amplify? if contribution ratio rises above 0.5 within a few thousand
steps, the two-CTM hypothesis is confirmed.

## Qwen3.5-4B — failed experiment, VRAM killed multi-tick (2026-03-15)

tried scaling up backbone from 0.6B to 4B. the idea: bigger backbone = better features for
CTM to work with. reality: 32GB VRAM on 5090 forced `--no-multi-tick`, which defeats the
entire purpose of CTM.

### what we ran

```
Qwen/Qwen3.5-4B, K=32, --no-multi-tick, --ctm-layers last
seq_len=1024 (halved from 2048), batch=1, grad_accum=4
resumed from step 6000, ran to ~8850/15000
VRAM: 29.4/32.6 GB (no room for multi-tick)
speed: ~9s/step, ~450 tok/sec
```

### why it failed

**no multi-tick = no continuous thought.** without multi-tick, the CTM gets one iteration per
token — it's just a weird MLP replacement, not a thinking machine. the results confirm this:

- **loss oscillating**: bounced between 2.4-3.0 from step 7000-8850, never converging
- **plasticity dead**: delta flat at 0.42-0.44% across all rehearsals (steps 8000-8800)
- **generation broken**: "The capital of France is **__________**" — outputting fill-in-the-blank
- **tick t0 dominates at 30-42%**: CTM barely iterating, just passing through initial state
- **dream never converges**: delta 2-5, layer 31

### plasticity rehearsal history

| step | delta | loss |
|------|-------|------|
| 8000 | 0.42% | 2.19 |
| 8100 | 0.43% | 2.27 |
| 8200 | 0.43% | 2.69 |
| 8300 | 0.42% | 3.45 |
| 8400 | 0.44% | 2.45 |
| 8500 | 0.44% | 2.51 |
| 8600 | 0.43% | 2.81 |
| 8700 | 0.44% | 2.41 |
| 8800 | 0.44% | 2.95 |

### lesson learned

**bigger backbone ≠ better CTM if you can't afford multi-tick.** CTM's value IS the iterative
thinking. without it, the model has no mechanism for continuous thought — it's just an
expensive linear layer. should have run K=16 with multi-tick instead of K=32 without it.

on 32GB VRAM with a 4B backbone:
- K=32 no-multi-tick: fits but useless (what we did)
- K=16 multi-tick: might have fit, would have been meaningful
- K=32 multi-tick: OOM

the 0.6B backbone remains the right scale for 32GB consumer GPUs with full CTM features.
or: get a bigger GPU.

## conclusions — what we know and the path forward (2026-03-15)

after weeks of experiments across 8 training runs, 4 backbones (GPT-2 d12, Qwen2.5-0.5B,
Qwen3-0.6B, Qwen3.5-4B), and countless dead ends, here's what we actually know.

### proven facts

1. **plasticity works.** Qwen2.5-0.5B + CTM K=32 + multi-tick achieved 4/4 fact recall after
   compact_memory(). this is not a fluke — recall-aware compact with lr=3e-4, steps=30,
   recall_weight=0.7 reliably teaches and retrieves facts across conversation restarts.

2. **multi-tick is non-negotiable.** without it, CTM is just a linear layer. the 4B run with
   `--no-multi-tick` proved this definitively: loss oscillates, plasticity delta flatlines at
   0.42%, generation outputs fill-in-the-blank blanks. one iteration ≠ continuous thought.

3. **CTM must replace FFN, not sit alongside it.** additive mode (MLP + CTM) lets the frozen
   FFN dominate. the CTM gets squeezed to rank 3 because training has no incentive to route
   signal through it. replacement mode forces all signal through CTM — that's what worked.

4. **frozen backbone is necessary.** unfreezing backbone layers lets the FFN absorb all
   gradient — it has more parameters, better initialization, and pretrained momentum. the CTM
   starves. keep the backbone frozen, force learning into the CTM.

5. **stronger backbone = harder for CTM.** Qwen2.5-0.5B worked because the backbone was weak
   enough that CTM had room to contribute. Qwen3-0.6B's c_proj collapsed to rank 3 — the
   backbone explained the data too well. this is the fundamental tension.

### the VRAM wall

CTM's compute cost scales with K (iterations) × sequence length × batch size. multi-tick
training multiplies this further (K forward passes for loss). on 32GB:

| backbone | K | multi-tick | VRAM | works? |
|----------|---|-----------|------|--------|
| 0.5B | 32 | yes | ~16 GB | **YES — plasticity proven** |
| 0.6B | 32 | yes | ~16 GB | trains but c_proj bottleneck |
| 2B | 32 | no | ~20 GB | FFN dominates, no plasticity |
| 4B | 32 | no | ~29 GB | dead — no multi-tick, no thought |
| 4B | 16 | yes | ~25 GB? | **untested — might work** |
| 4B | 32 | yes | OOM | needs H100/H200 |

### the path forward

**immediate (5090 32GB):**
- reproduce Qwen2.5-0.5B plasticity cleanly as the reference checkpoint
- try 4B + K=8 or K=16 with multi-tick — might fit, and lower K means faster search
- try multi-CTM layers on 0.5B (2-3 layers fit at batch=1)
- build autosweep script: backbone × K × num_ctm_layers, auto-kill on plateau

**with funding (H100 80GB / H200 140GB):**
- 4B backbone + K=32 + multi-tick + multi-layer CTM — the real experiment
- systematic hyperparameter sweep in parallel (dozens of runs, not one-at-a-time)
- the actual product needs a backbone strong enough to be useful (≥2B)

**open questions:**
- does lower K (8-16) preserve plasticity? the 0.5B success was at K=32 but nobody tested less
- how many CTM layers are needed on bigger backbones? one might not be enough to override
  30+ frozen FFN layers
- can spectral regularization actually fix the c_proj rank bottleneck given enough steps?
- is there a backbone sweet spot between "too weak to be useful" and "too strong for CTM"?

### the bottom line

we have proof of concept. plasticity — teaching an LLM new facts at inference time through
iterative internal thought — is real and works. what we don't have is the scale. the 0.5B
model that works is too small to be useful. the bigger models that would be useful can't fit
the full CTM pipeline on consumer hardware.

this is now a resource problem, not a research problem. the algorithm works. we need GPUs
to find the right configuration at useful scale, and that means either bigger hardware or
a systematic sweep that's smarter about exploring the space.

## Qwen2.5-0.5B K=4 + bound-guided aux — 8GB AMD GPU (2026-03-26)

first run with Angeris bound diagnostics and bound-guided auxiliary supervision.
Qwen2.5-0.5B backbone, K=4, on RX 7600M XT (8GB VRAM). the diagnostics changed everything.

### what worked

- **bound diagnostics caught bottlenecks in real-time**: c_proj rank expanded 1→49→62→149
  over 6000 steps. condition number dropped from 1.4M to 3.7K. would have caught the Qwen3
  rank-3 collapse at step 50 instead of step 11,000.
- **chunked multi-tick loss**: 248K vocab on 8GB — each tick's logits = 1GB. chunked lm_head
  into 256MB pieces + gradient checkpointing. no OOM.
- **plasticity works at step 2000**: compact_memory(lr=3e-4, steps=30) taught "Zyphrax→purple",
  4/4 recall. on an 8GB consumer GPU with K=4 ticks.
- **generation quality**: "capital of France → Paris" correct, coherent stories, fibonacci code.

### critical finding: plasticity has a critical period

| checkpoint | compact loss | recall "purple" | c_proj rank90 |
|-----------|-------------|-----------------|---------------|
| step 2000 | 11.4→8.1 | **YES** (4/4) | 62 |
| step 3000 | 12.8→8.9 | NO | ~80 |
| step 5000 | 12.3→9.1 | NO | 123 |

**step 2000 = plastic. step 3000+ = solidified.** same compact settings, same lr, same
steps. the weights won't budge after ~3000 steps. verified: not a CPU/GPU or dtype issue
(tested fp32 on GPU, same failure).

the optimization landscape hardens as training progresses. like biological critical periods:
young brains (early training) are plastic, older brains have solidified pathways. the model
needs plasticity exercises DURING training to maintain malleability — `--plasticity-every 25`
runs mini compact_memory every 25 steps to keep the weights in a plastic-friendly region.

### sequential compact: catastrophic forgetting

taught 5 facts in sequence. bounds stayed healthy (rank 62→65, no dead neurons, condition
stable). but each compact overwrites the previous — fact #2 erases fact #1. the plastic
LoRA pathway (plastic_A, plastic_B) has zero norm despite gate=0.5 — compact writes to
main weights, not the adapter. fix needed: null-space projection so each fact writes to
an orthogonal direction. rank 65 = room for 65 independent facts, they just need to not
step on each other.

### val loss mismatch (train/eval protocol)

val loss went UP (4.5→5.8) while train loss went DOWN (3.0→2.4). not overfitting — the
eval uses final-tick loss but training optimizes argmin across all K ticks. when ticks
specialize (tick 0 wins 52% of tokens), final-tick loss rises while the model IS learning.
fixed eval to report both final-tick and argmin-tick val loss.

## dual-CTM experiment — hippocampus + neocortex (2026-03-26)

implemented separate memory CTM (hippocampus) and cognitive CTM (neocortex) in the same
model. the idea: neocortex handles language (stable, can solidify), hippocampus handles
memory (kept plastic via compact_memory).

- cognitive CTM: replaces last FFN in backbone (existing behavior)
- memory CTM: extra layer after backbone norm, before lm_head
- compact_memory targets hippocampus exclusively
- both train on language loss but serve different purposes

tested on Qwen2.5-0.5B + K=4, 8GB AMD GPU. VRAM: 52% (fits). ran to step ~200 before
killing — loss 2.9 at 2.8s/step.

### why this doesn't solve the problem

the frozen backbone constrains BOTH CTMs equally. training on language modeling loss
causes BOTH to solidify around the same task. the hippocampus label is cosmetic — there's
no training signal that teaches it to be a memory module vs a language module. it will
hit the same critical period as the single CTM.

the real issue: compact_memory's 30-step gradient descent and training's 10k-step gradient
descent compete for the same weight space. training always wins.

## the synthesis — what the full training log teaches (2026-03-27)

reading the entire log from step 0 to here reveals a pattern. every failed experiment
shares one root cause, every success confirms the same principle.

### the pattern of failures

1. **12 CTMs (d12, K=3)**: 11 untrained CTMs corrupt signal → fix: single CTM
2. **single CTM K=1**: generation works, plasticity fails (1 tick = no convergence) → fix: more ticks
3. **K-ramp 1→2**: generation breaks (new tick injects noise) → fix: train at target K from start
4. **isis warm-start**: FFN patterns baked in, CTM generates garbage → fix: train from scratch
5. **frozen Qwen + CTM**: plasticity works at step 2000, dies by step 3000 → fix: ???
6. **frozen Qwen + dual CTM**: same problem, frozen backbone constrains both CTMs

every fix creates the next problem. we're patching a leaky boat.

### the invariant across all successes

- **d12 from scratch** produced the best generation quality (coherent paragraphs at step 5k)
- **single CTM with cache-aware from step 0** produced working CTMCache generation
- **Qwen2.5-0.5B at step 2000** (young model) produced working plasticity (4/4 recall)
- **K at target from step 0** avoided all K-ramp disasters

every success had: from-scratch or young weights + single CTM + cache-aware + target K.

### the root cause

compact_memory's gradient descent and training's gradient descent compete for the same
weight space. training pushes weights toward language modeling. compact pushes toward
fact recall. whoever runs longer wins. training runs for 10k steps. compact runs for 30.
training always wins.

separate modules (dual CTM) don't help if the training objective is the same. the
hippocampus solidifies around language modeling just like the neocortex.

### the solution: plasticity-preserving training

train from scratch — d12 + single CTM at target K from step 0 — with a dual-objective
training loop:

1. **70% of steps**: normal language modeling loss (learn to speak)
2. **30% of steps**: plasticity exercise — pick a random fact from training data, compact
   it, verify recall, continue training. the loss includes "can I still compact?"

this is weight training with stretching. the model learns to STAY PLASTIC while learning
language. the plasticity exercises during training prevent weights from solidifying into a
rigid minimum that rejects future compact_memory calls.

### what we built this session (2026-03-26/27)

code contributions:
- **Angeris bound diagnostics** in dream(): c_proj SVD rank/condition, synapse utilization,
  neuron health, bottleneck identification. caught rank-1 collapse at step 50 instead of 11k.
- **bound-guided aux loss**: per-tick supervision weight ∝ synapse gap from optimality.
  connects Angeris SDP framework with deep supervision theory.
- **chunked multi-tick loss**: 248K vocab on 8GB — chunked lm_head into 256MB pieces +
  gradient checkpointing. no OOM on consumer GPU.
- **dual eval**: final-tick + argmin-tick val loss. exposed the train/eval protocol mismatch.
- **dual-CTM architecture**: hippocampus + neocortex. works but doesn't solve root cause.
- **compact_memory cache fix**: rebuild cache each step to avoid backward graph leak.
- **plastic LoRA dtype fix**: bf16/fp32 mismatch in plastic pathway.

key findings:
- c_proj rank expanded 1→149 over 6000 steps on Qwen2.5-0.5B (healthy)
- plasticity has a **critical period**: works at step 2000, dead by step 3000
- sequential compact causes **catastrophic forgetting** — each fact overwrites previous
- plastic LoRA pathway (plastic_A/B) has zero norm — compact writes to main weights
- val loss mismatch is train/eval protocol difference, not overfitting
- 4B backbone without multi-tick = useless (CTM is a passthrough)

### the plan: d12 from scratch with plasticity exercises

the rational path forward:

```
d12, single CTM, K=4 or K=8 from step 0
cache-aware 0.30 from step 0
bound-guided aux from step 0
plasticity exercises every 25 steps during training
Angeris diagnostics every 50 steps
train on FineWeb/climbmix
8GB AMD GPU (d12 fits easily)
```

d12 won't be GPT-4. but if plasticity survives 10k+ steps of training with interleaved
exercises, that's the real breakthrough — a model that never stops learning.

## gradient-free neuroplasticity — the breakthrough (2026-04-03)

### what happened

we proved that a language model can learn new facts and recall them perfectly
without a single gradient computation. 12 facts, 3 plural alters, 100% recall,
zero forgetting, instant storage and retrieval. no backprop, no optimizer, no
training loop.

### the architecture

```
Frozen Qwen 2.5-0.5B (24 layers, 494M params)
    ↓
layer 23 hidden state at last token = MEMORY KEY (896-dim)
    ↓
cosine similarity vs stored keys → GATE (0 or 1)
    ↓
sequential logit biases per answer token → INJECT into output
    ↓
final logits = base Qwen logits + gated memory logits
```

no CTM needed for the core mechanism. the frozen backbone's own hidden states
are perfect memory keys — causal attention GUARANTEES identical hidden states
for identical prefixes. store once, match exactly, inject answer tokens.

### the path that got us here

1. **CTM v2 with 4 brain regions** (input/attention/output/motor) — built and validated
   on QEC. 94.9% accuracy, 30% fewer params than v1 flat. per-region tick counts
   matching brain oscillation timescales. but sync discrimination was the bottleneck.

2. **Angeris bounds diagnosis** — computed per-region per-tick gaps. found all
   inter-region synapses are ALREADY at least-squares optimum (gap ≈ 0). the bottleneck
   was never the weights — it was the architecture (where to inject the signal).

3. **residual stream suppression** — RMSNorm after 24 layers mathematically suppresses
   any additive delta in the residual stream. one layer can't override 23 others.
   inject at LOGITS, not residual stream. the hippocampus projects directly to output.

4. **least-squares language learning** — one solve on c_proj mapped sync → MLP output
   at 99.97% reconstruction. loss 3.2 → 1.6, zero gradients. proved CTM sync contains
   enough information for language.

5. **sync discrimination failure** — trained CTM in replacement mode (2000-5000 steps),
   sync patterns still 0.99+ cosine similar across inputs. the sync accumulator with
   r≈1 converges to a fixed point regardless of input (law of large numbers).

6. **instant sync (r≈0)** — set decay to max, only last tick's pairwise product matters.
   dropped cosine sim from 0.994 to 0.79. but still not enough for reliable gating.

7. **key insight: bypass sync entirely** — the backbone's own hidden states at the
   prompt's last token are PERFECTLY discriminative (self-match = 1.000, cross-match
   = 0.37-0.52). causal attention guarantees this. no CTM needed for the KEY.

8. **sequential logit injection** — store one logit bias per answer subword token.
   at generation step 0: boost "Z". step 1: boost "y". step 2: boost "ph". step 3: boost "rax".
   stops after answer length. base Qwen handles the rest.

### results

```
12 facts stored across 3 alters (spy, scientist, storyteller)
12/12 recalled correctly (100%)
0/3 non-taught prompts triggered (0% false positive on dissimilar)
1/1 structurally similar prompt cross-activated (Aurora vs Beta, 0.954 sim)
base Qwen knowledge preserved (Paris, ML, etc.)
```

### what this means

1. **no critical period** — memory is a separate key-value store. doesn't touch
   trained weights. works at step 1 or step 1,000,000 identically.

2. **no catastrophic forgetting** — each fact is an independent slot. teaching
   fact #12 doesn't affect fact #1. per-alter banks add further isolation.

3. **no gradients anywhere** — store = one forward pass + cosine key + logit bias.
   recall = one cosine match + logit injection. the Angeris framework told us
   WHERE to intervene (logits, not residual stream) and that synapse optimization
   was already done (gap ≈ 0).

4. **plural self-states** — different alters know different things. spy knows codes,
   scientist knows formulas, storyteller knows lore. automatic routing via cosine
   similarity. no cross-contamination.

### remaining issues

- **entity specificity**: "code for Aurora" vs "code for Beta" = 0.954 cosine sim.
  structurally similar prompts cross-activate. fixable with multi-position keys
  or learned key projections.

- **answer length**: currently must know answer token count at storage time.
  could auto-detect via confidence drop.

- **integration with CTM v2**: the KV memory works without CTM. the CTM v2 brain
  regions add value for adaptation under drift (QEC: +0.9% Hebbian) and for
  the from-scratch learning goal. the two systems are complementary.

### code

- `nanochat/memory_head.py` — low-rank logit projection (superseded by KV approach)
- `nanochat/plural.py` — plural self-state system with per-alter c_proj
- `nanochat/ctm_v2_block.py` — 4-region CTM block, drop-in for CTMBlock
- `test_kv_memory.py` — KV memory proof of concept (10/10 facts)
- `test_plural_kv.py` — plural KV system (12/12 facts, 3 alters)
- `models/ctm_v2.py` (CTM repo) — standalone v2 with factory function
- `tasks/qec/realistic_noise.py` (CTM repo) — PAEMS-calibrated noise model
- `tasks/qec/train_v2.py` (CTM repo) — v2 training on QEC

### the formula

```
plasticity = frozen_backbone_hidden_states (key)
           + cosine_similarity (gate)
           + sequential_logit_bias (value)
           + per_alter_kv_banks (plural)

no gradients. no backprop. no training loop. no critical period.
```
