# neural memory in continuous thought machines

how a language model remembers what it experienced.

## the core trade

standard transformers have one thinking step per token. the FFN fires once — a
single matrix multiply — and that's the model's entire thought. it's a lookup
table. fast, parallelizable, but fundamentally static. the weights encode
everything the model will ever know, frozen at the end of training.

CTM replaces the FFN with an iterative dynamical system. each token gets K ticks
of processing: cross-attend to input, run through a synapse network, update a
neural trace, accumulate synchronisation statistics. the model literally thinks
multiple times before committing to an output. this is slower than a single
matrix multiply. you lose parallel bruteforce. but you gain something the FFN
can never have: a continuous internal state that evolves over time and can be
written back into the weights.

that's the trade. speed for plasticity. the FFN is a camera — one snapshot per
token. the CTM is a brain — a dynamical system that accumulates experience.

## the three memory systems

biological brains have multiple memory systems operating at different timescales.
so does our CTM. they compose into a pipeline: experience → working memory →
permanent knowledge.

### 1. working memory: CTMCache (seconds)

during generation, each token's thinking starts where the previous token's
thinking ended. the CTMCache carries forward six quantities per layer:

```
state       (BT, D)        — current neuron activations
trace       (BT, D, M)     — rolling window of M recent states (neural trace)
alpha_out   (BT, n_synch)  — running sum of pairwise products (S_out)
beta_out    (BT, n_synch)  — running normalizer for S_out
alpha_act   (BT, n_synch)  — running sum of pairwise products (S_action)
beta_act    (BT, n_synch)  — running normalizer for S_action
```

this is stream of consciousness. the model reads token 47 and its internal state
still carries echoes of token 1. the sync accumulators are especially important —
they encode which neuron pairs have been co-firing consistently across all tokens
seen so far. this is the raw material for long-term memory formation.

the dopamine field gates how strongly each token's activity contributes to the
sync accumulators. surprising tokens (high prediction error) get dopamine > 1,
amplifying their contribution. boring tokens get dopamine < 1. the model
"remembers harder" when surprised, just like the brain's dopaminergic system
gates synaptic plasticity.

```python
# dopamine from prediction surprise (engine.py)
token_surprise = -log_prob(sampled_token)
dopamine = 1.0 + tanh(surprise - running_mean)  # range [0, 2]
ctm_cache.dopamine = dopamine

# dopamine gates sync accumulation (gpt.py, CTMBlock.forward)
pp_out = left * right * dopamine
alpha_out = r * alpha_out + pp_out
beta_out = r * beta_out + dopamine
```

### 2. long-term memory: compact_memory (permanent)

after a generation session, the accumulated sync patterns can be written into the
model's synapse weights permanently. this is compact_memory() — the analog of
hippocampal-cortical memory transfer.

three mechanisms:

**sync-driven hebbian update.** the accumulated S_out synchronisation
(alpha/sqrt(beta)) encodes which neuron pairs co-activated during the session.
compare this to the baseline sync that start_state would produce with no input.
the difference is what the model learned from THIS experience.

```python
synch_accumulated = alpha_out / sqrt(beta_out)     # what happened
synch_baseline = pp_baseline * sqrt(beta_baseline)  # what would happen with no input
sync_delta = synch_accumulated.mean(0) - synch_baseline  # what's new
```

**novelty gating.** not every sync pattern is worth remembering. we gate by
novelty — only update where the sync diverged most from baseline. the top 50%
most novel pairings get plasticity, the bottom 50% don't. this prevents the
model from overwriting existing knowledge with redundant information.

```python
novelty = sync_delta.abs()
gate = (novelty > novelty.median()).float()
gated_delta = sync_delta * gate
```

**homeostasis.** unconstrained hebbian learning causes runaway weight growth.
after each update, we clamp every modified weight tensor to at most 1% norm
growth. the direction changes, but the magnitude stays stable. this is analogous
to synaptic scaling — the brain's mechanism for keeping total synaptic strength
in a healthy range.

```python
max_norm = base_norm * 1.01
if current_norm > max_norm:
    weight *= max_norm / current_norm
```

the update targets two weight matrices: c_proj (the readout layer that maps sync
→ output) and the last synapse up-layer. c_proj learns to amplify the sync
channels that carried novel information. the synapse layer learns the state
dynamics that produced those patterns.

### 3. consolidation during sleep (minutes)

every N training steps, the model "sleeps": dreaming, compacting, consolidating.

**dreaming.** run inference on recent sequences, collecting per-layer convergence
diagnostics. which layers are still actively changing their state? which have
settled? this tells us where the model is still "thinking hard" vs "confident."

**compaction.** run compact_memory on the dream sequences — write the sync
patterns from dreaming into permanent weights.

**consolidation.** self-distillation on a replay buffer. a frozen snapshot of
the model (the "teacher") provides target logits. the live model trains only its
synapse and NLM parameters to match. this anchors the plastic parameters back
toward the training distribution, preventing catastrophic forgetting.

```
step 9010 sleep cycle output:
  Dreaming about 4 sequences (losses: 3.44, 3.80, 3.93, 3.15)
  Layer  0: K 2->2 [20.96 -> 25.19] active
  Layer 11: K 2->2 [249.13 -> 760.66] active
  Consolidation: 16 steps, loss=0.7171, certainty=0.6784
  Replay buffer: 10 entries, loss range [3.11, 3.93]
```

## the missing piece: cache-aware training

here's the problem. during training, every sequence starts with a blank CTMCache.
all tokens are processed in parallel with fresh state. the model has never seen
what accumulated cache state looks like.

at inference, CTMCache accumulates over 50+ tokens. the sync statistics build up,
the trace fills with real history, the state carries forward tick-to-tick. the
model encounters activation patterns it was never trained on. it produces garbage.

this is the fundamental blocker for neuroplasticity. compact_memory can write
patterns into weights, but the model can't READ its own modified state, because
it was trained to expect blank state at every token boundary.

### the fix: cache-aware training

30% of training micro-steps now work differently:

```
normal step (70%):
  [token 0 .................. token 1023]
  fresh CTMCache → compute loss on all 1024 tokens

cache-aware step (30%):
  [token 0 ....... token 511] | [token 512 ....... token 1023]
  fresh cache → forward (no grad) | carry cache → compute loss HERE
       "experiencing"                    "learning with memory"
```

the first half is pure experience — the model reads text, its internal state
evolves, sync accumulators build up naturally. no gradients, no learning. just
accumulation.

the second half is learning with accumulated state. the model has to predict
tokens while carrying all that history. gradients flow through the second half
only. the model learns: this accumulated state is useful information, not noise.

this is how the model learns to read its own CTMCache. once it can do that:

1. generation works with CTMCache (stream of consciousness across tokens)
2. compact_memory writes patterns the model can actually use
3. the model gets better at things it's practiced, without gradient descent
4. the full loop closes: experience → memory → permanent knowledge

### what we observe

the model shows a characteristic loss spike when cache-aware training begins —
it's encountering accumulated state for the first time. the spike settles as the
model adapts:

```
step 9000: loss 3.29  (resume point, fresh cache-aware training)
step 9005: loss 3.49  (spike — unfamiliar accumulated state)
step 9011: loss 3.38  (settling — learning to use cache)
```

## the full pipeline

putting it all together, here's what happens during a single generation session:

```
1. prompt arrives
   → fresh CTMCache created (or loaded from episodic memory, future work)

2. prefill: process all prompt tokens
   → KVCache fills (attention memory)
   → CTMCache accumulates (thinking memory)
   → each token's K ticks build sync statistics
   → dopamine gates contribution by prediction surprise

3. generate: produce tokens one at a time
   → each token starts where previous ended (stream of consciousness)
   → sync accumulators encode running co-activation history
   → surprising tokens get higher dopamine → stronger sync contribution
   → CTMCache grows richer with each token

4. compact: write experience into weights (optional)
   → compare accumulated sync to baseline
   → gate by novelty (top 50% most divergent)
   → hebbian update to c_proj + last synapse layer
   → homeostasis: clamp to 1% norm growth

5. next session: the model is different
   → modified weights produce new activation patterns
   → patterns the model encountered before are handled faster/better
   → new patterns still get full processing
   → no gradient descent required — pure inference-time learning
```

and during training, the sleep cycle runs periodically:

```
every 10 steps:
  1. dream()  — run inference, collect per-layer diagnostics
  2. compact_memory() — write dream patterns into weights
  3. consolidate() — self-distill on replay buffer (prevent forgetting)
```

## the generate_and_compact API

for inference-time plasticity, a single call does everything:

```python
engine = Engine(model, tokenizer)

# generate AND write experience into permanent weights
results, masks, plasticity_stats = engine.generate_and_compact(
    tokens=prompt_tokens,
    num_samples=1,
    max_tokens=256,
    plasticity_lr=1e-5,  # how aggressively to learn
)

# the model is now different — it remembers this conversation
# next generation session benefits from this experience
```

for persistent conversations with continuous memory:

```python
session = Session(model, tokenizer, max_seq_len=4096)
reply1 = session.say("hello, who are you?")
reply2 = session.say("what did i just ask you?")  # model remembers via KV cache
# CTMCache carries forward thinking state between turns (once cache-aware trained)
```

## what this is not

**not fine-tuning.** fine-tuning requires a loss function, backpropagation, and an
optimizer. compact_memory uses hebbian learning — coincidence detection between
neuron pairs, gated by novelty and dopamine. no gradients. no optimizer state. no
training data format. just: which neurons co-fired during this experience?

**not RAG.** retrieval-augmented generation stores text chunks and retrieves them
into the context window. our memory is sub-symbolic — sync statistics, not tokens.
the model doesn't "look up" past experience. past experience literally changes the
weights that process future input.

**not prompt caching.** prompt caching stores KV pairs to avoid re-computing
attention on repeated prefixes. that's about speed. our CTMCache carries thinking
state — the accumulated sync dynamics from processing previous tokens. it's not
about avoiding re-computation, it's about continuity of thought.

## the bet

the transformer FFN is a lookup table. one matrix multiply, one answer. it knows
what it was trained on and nothing else. every inference session is amnesia.

the CTM is a dynamical system with memory. it accumulates experience into sync
statistics, gates by surprise, writes patterns into weights, and gets better at
things it's practiced — all at inference time, without gradient descent.

a chess novice consciously traces every move. a grandmaster sees the board and the
move appears. not because they think faster, but because years of practice compiled
conscious reasoning into automatic pattern recognition. compact_memory is that
compilation step — conscious processing (high K, many ticks) becoming automatic
knowledge (modified weights that produce the right dynamics at K=1).

whether this works at 469M params on one GPU is the experiment. the machinery
exists. cache-aware training is running. the loop is closing.
