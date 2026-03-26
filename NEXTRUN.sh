#!/bin/bash
# Next experiment: K=8 with bound-guided auxiliary supervision
#
# Hypothesis: K=8 + aux per-tick supervision matches K=32 plasticity
# at half the VRAM, because every tick is forced to improve predictions.
#
# From poker CTM findings:
#   - Without aux: later ticks diverge (loss 354K)
#   - With aux: monotonic improvement (0.97 → 0.75)
#   - Spectral norm on step weights prevents divergence at root cause
#   - KL adaptive halt saves compute on easy tokens
#
# Changes from last successful run (qwen25_ctm_k32_v1):
#   - K=8 instead of K=32 (save ~50% VRAM)
#   - ctm-aux-weight=0.1 (auxiliary per-tick supervision)
#   - cache-aware from step 0
#   - multi-tick enabled (mandatory)
#
# Expected VRAM: ~10 GB (vs ~16 GB for K=32)
# Expected speed: ~2x faster per step

python3 -m scripts.train_qwen_ctm \
  --run qwen25_ctm_k8_aux \
  --backbone Qwen/Qwen2.5-0.5B \
  --ctm-iterations 8 \
  --ctm-aux-weight 0.1 \
  --ctm-memory-length 16 \
  --ctm-memory-hidden 32 \
  --ctm-synapse-depth 32 \
  --cache-aware-ratio 0.30 \
  --num-iterations 15000 \
  --device-batch-size 4 \
  --max-seq-len 2048 \
  --lr 1e-3 \
  --warmdown-ratio 0.3 \
  --eval-every 250 \
  --save-every 500 \
  --sample-every 100

# Comparison run: same but K=32, no aux (baseline, to measure improvement)
# python3 -m scripts.train_qwen_ctm \
#   --run qwen25_ctm_k32_baseline \
#   --backbone Qwen/Qwen2.5-0.5B \
#   --ctm-iterations 32 \
#   --ctm-aux-weight 0.0 \
#   --cache-aware-ratio 0.30 \
#   --num-iterations 15000 \
#   --device-batch-size 2 \
#   --max-seq-len 2048
