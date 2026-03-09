# training log

a record of our model's journey from noise to something that thinks.

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

plan: let FFN run to 10k steps (~46 min total), then warm-start CTM from
the FFN checkpoint. the CTM blocks start fresh but attention + embeddings
get a 10k-step head start. compare this warm-started CTM against our
from-scratch CTM to measure the benefit.

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
