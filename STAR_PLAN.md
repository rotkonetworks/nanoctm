# self-taught reasoning in continuous thought machines

a technical plan for adaptive compute allocation and experience-driven reasoning improvement.

## the core observation

CTM already has the machinery for self-taught reasoning. it just doesn't know it yet.

standard self-taught reasoning (STaR, Zelikman et al. 2022) works in token space:
generate chain-of-thought, check answer, train on winners. expensive — the model
has to verbalize every reasoning step, and you're bottlenecked by vocabulary and
sequence length.

CTM reasons in latent space. each of K ticks is a refinement step — cross-attend
to input, process through synapse, update trace, accumulate sync. the model
literally thinks multiple times before committing to a token. tick 0 is gut
instinct. tick K-1 is considered judgment. the certainty signal tells you which
tick each token trusted most.

the insight: we don't need chain-of-thought tokens. the ticks ARE the chain of
thought. self-taught reasoning becomes self-taught tick allocation — learn when
to think hard, when to trust your gut, and make that knowledge permanent.

## what we already have

before building anything new, inventory what exists:

**multi-tick loss with per-token routing** (gpt.py ~line 998):
- at each tick k, we compute CE loss and certainty per token
- `argmin(loss)` picks the best-performing tick per token
- `argmax(certainty)` picks the most confident tick per token
- combined: 0.5 * L_argmin + 0.5 * L_argmax_certainty
- this already trains the model to route easy tokens to early ticks

**state delta tracking** (gpt.py ~line 478):
- `delta = ||state_k - state_{k-1}||` measured every tick
- dream() exposes this per-layer: fast-converging layers vs slow layers
- existing adaptive stub (line ~507): break if delta increases

**sync accumulators** (gpt.py ~line 488):
- `alpha_out`, `beta_out` track pairwise co-activation across ticks
- these encode HOW the model thought, not just WHAT it concluded
- already used by compact_memory() for Hebbian plasticity

**compact_memory** (gpt.py ~line 1340):
- compares accumulated sync to baseline (start_state resting sync)
- gates by novelty (top 50% most divergent pairings)
- writes into c_proj + last synapse up-layer
- homeostasis: 1% max norm growth per call

**CTMCache** (gpt.py ~line 211):
- persistent state/trace/sync across tokens at inference
- dopamine field for neuromodulatory gating
- stream of consciousness — each token starts where previous ended

none of this is wired together yet. the pieces exist in isolation. the plan
below connects them into a closed loop.

---

## phase 1: per-token adaptive K at inference

**what**: let each token exit the tick loop early when it's confident enough.
currently all tokens run K ticks. this wastes compute on easy predictions
("the", "is", "of") that don't benefit from deep thought.

**how**: after each tick, compute certainty from the sync readout. if certainty
exceeds a threshold, freeze that token's output and let it skip remaining ticks.

```python
# CTMBlock.forward, inside the tick loop
for k in range(K):
    # ... existing: cross-attend, synapse, trace, NLM, sync update ...

    # new: per-token early exit (inference only, not training)
    if adaptive and k >= 1 and not self.training:
        synch_k = alpha_out / torch.sqrt(beta_out + 1e-8)
        logits_k = self.c_proj(synch_k.to(dtype))

        # certainty from entropy
        probs = F.softmax(logits_k, dim=-1)
        ent = -(probs * (probs + 1e-10).log()).sum(-1)
        cert = 1.0 - ent / math.log(probs.shape[-1])

        newly_done = (~exited) & (cert > exit_threshold)
        if newly_done.any():
            final_out[newly_done] = logits_k[newly_done]
            exit_tick[newly_done] = k
            exited |= newly_done

        if exited.all():
            break
```

**complexity**: ~50 lines in CTMBlock.forward. no new parameters. no training
changes. pure inference optimization.

**expected savings**: 30-50% compute at inference. easy tokens (function words,
common continuations) exit at K=1. hard tokens (factual recall, reasoning,
ambiguous context) run full K. the savings come from the long tail — most tokens
in natural language are predictable.

**measurement**: log exit_tick histogram. healthy distribution looks like
[40% K=1, 25% K=2, 20% K=3, 15% K=4]. if >80% exit at K=1, the threshold is
too low (model is lazy). if >80% run full K, threshold is too high (no savings).

**prerequisite**: a model trained at K>=2 with multi-tick loss. K=1 models have
nothing to adapt — there's only one tick.

---

## phase 2: tick predictor head

**what**: teach the model to predict, from its internal state at tick k, whether
more ticks will help. this replaces the heuristic certainty threshold from phase 1
with a learned routing decision.

**how**: small linear head on the sync signal, trained with supervision from the
multi-tick loss.

```python
# CTMBlock.__init__
self.tick_predictor = nn.Linear(self.n_synch, 1)  # predict: should I keep thinking?

# During multi_tick_loss(), after computing all_losses[K, B*T]:
# find optimal K per token — the first tick that's "good enough"
best_loss = all_losses.min(dim=0).values              # (B*T,)
good_enough = all_losses < (best_loss + 0.1)           # (K, B*T) bool
optimal_k = good_enough.float().argmax(dim=0)           # (B*T,) first good tick

# At each tick k during forward, predict "should I continue?"
for k in range(K):
    # ... existing tick computation ...
    synch_k = alpha_out / torch.sqrt(beta_out + 1e-8)
    should_continue = (optimal_k > k).float()           # 1 if more ticks needed
    pred = self.tick_predictor(synch_k).squeeze(-1)      # (BT,)
    tick_loss += F.binary_cross_entropy_with_logits(pred, should_continue)

# add to total loss
task_loss += 0.05 * tick_loss / K
```

**why this works**: the model already computes per-tick losses. we're just using
that signal to teach a tiny head to predict "is more thinking worth it?" the
head sees the sync state — the same signal that encodes certainty — and learns
the boundary between "confident enough" and "still confused."

**complexity**: one nn.Linear(384, 1) per CTMBlock = 385 params per layer.
negligible. the training signal is free — we already compute all_losses.

**interaction with phase 1**: at inference, replace the certainty threshold with
`tick_predictor(synch_k) < 0` (sigmoid < 0.5 = stop). learned routing instead
of hand-tuned threshold.

---

## phase 3: self-taught reasoning from verification

**what**: generate multiple solutions, verify which are correct, reinforce the
tick allocation that led to correct answers. this is STaR adapted for latent
reasoning.

**how**: for problems with verifiable answers (math, factual questions, code
output), generate at multiple K values, compare results, create training signal.

```python
def star_step(model, engine, prompt, answer, K_options=[2, 4]):
    """one step of self-taught reasoning."""
    examples = []
    for K in K_options:
        # temporarily set model to this K
        model.set_active_k(K)
        results, masks = engine.generate_batch(prompt, num_samples=4, max_tokens=64)

        for seq in results:
            text = tokenizer.decode(seq)
            correct = verify(text, answer)  # exact match, fuzzy, or tool-checked
            examples.append({
                'tokens': seq,
                'K': K,
                'correct': correct,
            })

    # find the minimum K that solved it
    correct_examples = [e for e in examples if e['correct']]
    if not correct_examples:
        return None  # too hard, skip

    min_k = min(e['K'] for e in correct_examples)

    # training signal: the correct solution at minimum K
    winner = next(e for e in correct_examples if e['K'] == min_k)
    return winner['tokens'], min_k  # train on this, with tick_predictor target = min_k
```

**the key insight**: we're not training on chain-of-thought reasoning traces.
we're training the model to allocate the right amount of latent computation.
"this problem needs K=4" is the learned signal, not "first I add 7+8..."

**data requirements**: problems with checkable answers.
- math: "7 + 8 = " → check if output contains "15"
- facts: "the capital of france is" → check for "paris"
- code: "def add(a,b): " → execute and verify
- counting: "how many r's in strawberry" → tool-verify

start small. 1000 math problems, 500 factual QA. the model doesn't need to be
good at these — it needs to learn WHEN it needs more ticks.

**training loop**:
```
for problem, answer in dataset:
    winner_tokens, optimal_k = star_step(model, engine, problem, answer)
    if winner_tokens:
        # standard CE loss on winner_tokens
        # tick_predictor target: optimal_k for each position
        # this teaches: "math tokens need K=4, article tokens need K=1"
        train_on(winner_tokens, tick_target=optimal_k)
```

---

## phase 4: plasticity-backed reasoning (the endgame)

**what**: make self-taught reasoning improvements permanent through Hebbian
plasticity. the model encounters a hard problem type, figures out the right
tick allocation, and writes that knowledge into its synapse weights so it's
faster next time — without any gradient-based training.

**how**: extend compact_memory() to also update the tick_predictor bias based
on the sync patterns observed during successful reasoning.

```python
def reason_and_remember(model, engine, problem):
    """encounter → reason → verify → remember."""

    # 1. try with default adaptive K
    result, ctm_cache = engine.generate_with_cache(problem)

    if verify(result):
        # easy — model handled it. low dopamine, minimal plasticity.
        ctm_cache.dopamine = 0.3
        model.compact_memory(ctm_cache, lr=1e-6)  # gentle consolidation
        return result

    # 2. didn't work. escalate: retry with max K
    model.set_active_k(model.config.ctm_iterations)  # full K
    result2, ctm_cache2 = engine.generate_with_cache(problem)

    if verify(result2):
        # hard but solvable with more thought. high dopamine.
        ctm_cache2.dopamine = 2.0  # surprise! learn hard.
        model.compact_memory(ctm_cache2, lr=1e-4)  # aggressive write

        # the sync patterns from "thinking hard and succeeding" are now
        # written into synapse weights. next time similar input patterns
        # activate these synapses, the sync dynamics will naturally produce
        # states that the tick_predictor reads as "keep going."
        return result2

    # 3. still wrong. don't learn from failure (for now).
    return None
```

**why this might work**: compact_memory writes sync patterns — which neuron
pairs co-fired — into the readout weights. when the model "thinks hard" (high K,
many tick iterations), different neuron pairs co-fire than when it "thinks easy"
(low K, early exit). writing the "hard thinking" sync pattern into weights means
that next time similar input triggers those synapses, the internal dynamics
reproduce the "think harder" activation pattern — even at K=1, the sync signal
carries an echo of the deeper computation.

this is speculative. it depends on:
1. sync patterns being discriminative enough to encode "problem difficulty"
2. compact_memory's rank-1 updates being expressive enough to change routing
3. the model reading its own modified weights correctly (cache-aware training)

if even partially true, it means the model gets faster at things it's practiced.
first time seeing long division: K=4. tenth time: K=2, because the synapse
weights now encode the computation pattern. this is what skill acquisition
looks like — conscious effort becoming automatic.

---

## implementation order

| phase | effort | prereqs | what it proves |
|-------|--------|---------|----------------|
| 1. adaptive K inference | ~50 LOC | K>=2 model | compute savings, confirms tick utility |
| 2. tick predictor | ~80 LOC | phase 1 | model can learn metacognition |
| 3. STaR verification | ~200 LOC + data | phase 2, stable generation | self-improvement works |
| 4. plasticity reasoning | ~100 LOC | phase 3, cache-aware training | permanent learning from experience |

**critical path**: cache-aware training → stable generation → phases 1-4.
nothing works if the model can't handle its own accumulated state. this is
the same blocker as plasticity. solve it once, unlock everything.

total new code: ~430 lines. no architectural changes. no new modules. just
wiring together pieces that already exist: multi-tick loss provides the signal,
tick_predictor learns the routing, compact_memory makes it permanent.

## what this is NOT

this is not RLHF. there's no reward model, no PPO, no KL penalty. the model
doesn't learn what humans prefer — it learns what computation pattern works
for which input pattern. the reward signal is binary: did the answer verify?
the optimization target is compute allocation, not output distribution.

this is not chain-of-thought distillation. the reasoning happens in latent
tick space, not token space. there are no "let me think step by step" prompts.
the model's chain of thought is the sequence of sync states across K ticks —
invisible, continuous, and much cheaper than generating reasoning tokens.

this is not test-time compute scaling (the "just run more tokens" approach).
adaptive K scales compute PER TOKEN based on difficulty, not per problem by
generating more output. a 20-token answer to a hard math problem gets K=4 on
the computation tokens and K=1 on "the answer is". total overhead: ~2x on the
hard tokens, not 10x on the whole response.

## the bet

the transformer FFN is a lookup table. one matrix multiply, one answer.
the CTM tick loop is a dynamical system. it can orbit, converge, bifurcate.
fixed points of the dynamics are "thoughts" — stable patterns the system settles
into. different K means different time to settle. adaptive K means the system
finds its own fixed points at its own pace.

self-taught reasoning closes the loop: the system discovers which inputs need
deep orbits, which need shallow ones, and writes that knowledge into the
dynamics themselves through synaptic plasticity. it doesn't just solve
problems — it gets better at deciding how hard to think about them.

brains do this. a chess novice consciously traces every move. a grandmaster
sees the board and the move appears — not because they think faster, but
because years of practice compiled the conscious reasoning into automatic
pattern recognition. the conscious reasoning (high K) literally becomes
the intuition (low K) through synaptic consolidation.

that's what we're building. whether it works at 469M params on one GPU is
the experiment.
