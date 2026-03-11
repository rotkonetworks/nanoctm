# next run: what we learned and what to do differently

## the hard lessons

### 1. cache-aware training from step 0

the single biggest mistake: training 9000 steps without cache-aware, then bolting
it on. the model treated CTMCache as noise because it never needed it. 1000 steps
of gradual ramp couldn't undo 9000 steps of cache-ignorant learning.

the fix: `--cache-aware-ratio 0.30` from the moment CTM starts. not as a ramp,
not as an afterthought. the model must grow up with cache as part of its world.

### 2. the state expansion bug (FIXED)

**root cause of generation collapse.** during inference, the previous token's full
post-K-ticks state was inherited as the next token's initial state. the model
never trained with this — every training position starts from `start_state`.
result: out-of-distribution input at tick 0 → garbage cascading through all ticks
→ fixed attractor ("irusesCSscientist representationral boils").

**fix (implemented):** at inference, reset state and trace to `start_state` /
`start_trace`. carry ONLY sync accumulators (alpha/beta) — these are the memory
channel. training path unchanged (inherits everything, matching existing behavior).

the fix broke the attractor. cache now produces topic-aware, per-prompt-unique
output. quality still low but mechanism functions.

### 3. dopamine positive feedback loop (FIXED)

dopamine was range [0, 2]. garbage output → high surprise → dopamine > 1 →
harder sync accumulation on garbage → more garbage. self-reinforcing collapse.

**fix (implemented):** clamp to [0.5, 1.0]. never amplify above training
distribution (always 1.0 during training). break the feedback loop while
preserving dampening for predictable tokens.

### 4. don't change architecture mid-training

ve_gate_channels 12→32 corrupted the old model. K changes mid-run cause optimizer
state mismatches. learned the hard way: pick dimensions at init, never change them.

### 5. tick 2 degradation from cache noise

30% cache-aware steps inject unfamiliar state. tick 1 spends all compute absorbing
cache noise, tick 2 has nothing to work with. tick selection flipped — tick 1
outperformed tick 2 for hundreds of steps.

with cache-aware from step 0, the model learns to handle cache at tick 1 from the
start, leaving tick 2 free for actual reasoning.

### 6. loss floor from competing objectives

at 0.30 cache-aware ratio, 70% of steps optimize language, 30% fight cache noise.
they cancel out → flat loss at ~3.40 for 1000+ steps. the model isn't stuck, it's
balancing two tasks that pull in opposite directions.

with cache-aware from step 0, there's one objective: language WITH cache. no
competing pressures.

### 7. K=1 first, ramp K later

K=2 at the start means 2x slower training per step. at K=1, CTM is ~FFN speed
but WITH the cache/sync/trace machinery active. let the model learn language
efficiently at K=1, then add thinking ticks once the foundation is solid.

## the recipe

```
phase 1: FFN baseline (DONE — have 10k checkpoint, bpb 0.938)
  - pure transformer, no CTM. fast. establishes language fundamentals.
  - 5-10k steps on H200/H100.

phase 2: CTM K=1 with cache-aware from step 0
  - warm-start from FFN checkpoint
  - --cache-aware-ratio 0.30 from step 0
  - inference fixes active (fresh state, dopamine clamping)
  - fast training (~FFN speed), model learns cache as fundamental
  - target: bpb < 1.5, coherent generation WITH cache
  - save checkpoint every 500 steps

phase 3: bump K=2
  - resume from best K=1 checkpoint
  - model already knows cache, tick 2 has room to add value
  - 2x slower but each step is richer
  - target: tick 2 consistently lower loss than tick 1

phase 4: bump K=3+ (if K=2 saturates)
  - only when K=2 ticks are both productive
  - don't add ticks the model can't use

phase 5: plasticity testing
  - once generation with cache is coherent (bpb < 2.0 with cache)
  - teach wrong facts, compact_memory, test recall
  - success = any shift toward taught fact after compaction
```

## launch command (phase 2)

```bash
NANOCHAT_NO_OPTIM_COMPILE=1 python3 -u -m scripts.base_train \
  --use-ctm --depth=12 --ctm-iterations=1 --ctm-synapse-depth=32 \
  --device-batch-size=3 --total-batch-size=61440 --window-pattern=L \
  --model-tag=ctm_d12_k1_cache --run=ctm_k1_cache \
  --num-iterations=15000 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=500 \
  --sample-temperature=0.9 --save-every=500 --keep-checkpoints=5 \
  --cache-aware-ratio=0.30 \
  --scheduled-sampling=0.05 --elastic-weight=0
```

notes:
- `--ctm-iterations=1` — K=1, fast training
- `--cache-aware-ratio=0.30` — from step 0, no ramp needed
- `--run=ctm_k1_cache` — wandb tracking from start
- warm-start from FFN 10k: copy checkpoint, model will auto-detect MLP→CTM

## code changes needed before next run

already implemented (on branch `fix/cache-inference-invariants`):
- [x] fresh state per token at inference, carry sync only (gpt.py)
- [x] dopamine clamped to [0.5, 1.0] (engine.py)
- [x] adaptive cache-aware ramp (base_train.py) — keep for safety but shouldn't need

should add:
- [ ] separate logging of clean loss vs cache-aware loss (currently blended)
- [ ] tick diversity loss (small regularizer forcing ticks to specialize)
- [ ] wandb logging of per-tick selection over time
- [ ] cache-aware ratio defaults to 0.30 when --use-ctm is set

## hardware notes

- CTM at K=1-2 is ~2% MFU. GPU matmul power is wasted. cheaper GPU is fine.
- batch=1 works for CTM (sequential tick loop is the bottleneck, not batch size)
- H100 80GB at $1.27/hr > H200 140GB at $1.97/hr for this workload
- checkpoints: save to /dev/shm (RAM), rsync to local. don't use overlay disk.
- NANOCHAT_NO_OPTIM_COMPILE=1 always (muon hits recompile limit with K changes)
