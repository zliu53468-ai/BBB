# BBB PF 10D + XGBoost / Transformer Dual-Brain Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史
-> 256D / V23 Core
-> Core P(B)
-> fixed 7D features
```

Only the downstream correction layer changes.

## Architecture

```text
fixed 7D
  -> ShoeParticleFilter
  -> [pred_card_count, pred_banker_point, pred_player_point]
  -> fixed 7D + physical 3D = 10D

10D current row -----------------> XGBoost -> delta_xgb

last 10 rows of 10D
(left-zero padded + valid mask)
  -> 10D -> 32D input projection
  -> fixed sinusoidal position encoding
  -> 1-layer Multi-Head Attention
       num_heads = 2
       key_dim   = 16
       d_model   = 32
       dropout   = 0.30
  -> masked Global Average Pooling
  -> Dense(1)
  -> delta_transformer

delta_final =
    0.50 * delta_xgb
  + 0.50 * delta_transformer

delta_clipped = clip(delta_final, -0.10, +0.10)
final_p_B = clip(core_p_B + delta_clipped, 0, 1)
```

## Frozen 7D input

The original upstream feature vector remains exactly:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

The PF adds only:

8. `pred_card_count`
9. `pred_banker_point`
10. `pred_player_point`

## Shoe Particle Filter

The PF still maintains 1000 plausible latent eight-deck shoes and never claims
to know the hidden real cards.

Configuration:

```text
n_particles = 1000
R = 0.25
Q early = 0.005
Q late  = 0.020
ESS threshold = 500
```

The blind physical forecast remains side-effect-free and softly conditioned on
current Core P(B).

## Transformer temporal brain

Input shape:

```text
(batch, window_size=10, features=10)
```

For a shoe with fewer than 10 available rows, older missing rows are left-padded
with zeros. A valid-mask prevents those padding rows from participating in
attention or pooling.

A fixed sinusoidal position encoding is added before attention. This is required
so the attention layer can distinguish temporal order; without positional
information, attention followed by global averaging would not reliably know
which row came earlier or later.

Structure:

```text
Linear 10 -> 32
+ sinusoidal position encoding
MultiHeadAttention(num_heads=2, key_dim=16)
Dropout(0.30)
Masked Global Average Pooling
Dense 32 -> 1 residual
```

## XGBoost brain

```text
n_estimators = 65
learning_rate = 0.03
max_depth = 4
min_child_weight = 2.0
alpha = 0.05
lambda = 0.25
random_state = 42
```

## Causal training

Historical rows are replayed shoe by shoe.

For round t:

```text
physical_3d_t = PF forecast before outcome t
feature_10d_t = [fixed_7d_t, physical_3d_t]

Transformer window t =
    current feature_10d_t
    + previous up to 9 rows from the same shoe

target_t = actual_B_t - core_p_b_t
```

Only after row t is captured does outcome t update the PF for t+1.

## Validation

The same deterministic held-out shoe split is used for both models.

Validation reports:

```text
Core accuracy / Brier
XGBoost-corrected accuracy / Brier
Transformer-corrected accuracy / Brier
50:50 fused accuracy / Brier
```

Only the fused result is used by the acceptance gate.

The trainer does not assume that adding a Transformer improves prediction. If
held-out fused metrics fail the configured gate, the model is not exported
unless `--force` is explicitly used for diagnostics.

## Training

```bash
python -m pip install -r requirements-xgb.txt

python xgb_particle_filter_residual.py train \
  --input bgs_xgb_blind_physical_10d_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

Transformer defaults:

```text
epochs = 120
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001
early-stopping patience = 15
```

The best validation epoch is selected first. After validation passes, the
Transformer is retrained on all available rows for that selected epoch count.

## Portable browser inference

`transformer_residual.py` exports PyTorch weights to JSON.

Before writing the model bundle, export validation compares:

```text
PyTorch eval() output
vs
pure NumPy portable Transformer output
```

and refuses export if they differ beyond tolerance.

The browser runtime implements the same:

- 10 -> 32 projection
- sinusoidal position encoding
- Q/K/V projection
- two 16D attention heads
- padding mask
- output projection
- masked global average
- dense residual
- 50:50 fusion with XGBoost

The checked-in `residual_bias_model.json` remains `trained:false` until real
historical labeled rows are trained and pass validation.
