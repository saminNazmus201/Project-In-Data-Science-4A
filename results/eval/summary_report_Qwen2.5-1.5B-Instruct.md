# Summary level evaluation (Qwen/Qwen2.5-1.5B-Instruct)

87 hotel-aspect cases x 4 methods. NLI: cross-encoder/nli-deberta-v3-base. Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest. Mean with 95% bootstrap CI over the cases.

| Method | Has citation | Invalid citation | Faithful (NLI) | Contradicted | Citation precision | Polarity gap to pool | Polarity gap to evidence | Mentions a complaint | Minority lost | ROUGE-L | Words |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Top-K | 0.379 [0.276, 0.483] | 0.000 [0.000, 0.000] | 0.153 [0.119, 0.188] | 0.113 [0.076, 0.155] | 0.048 [0.000, 0.127] | 0.321 [0.271, 0.371] | 0.297 [0.255, 0.340] | 0.862 [0.782, 0.931] | 0.129 [0.059, 0.200] | 0.106 [0.100, 0.111] | 85 |
| MMR | 0.437 [0.333, 0.540] | 0.000 [0.000, 0.000] | 0.161 [0.123, 0.201] | 0.136 [0.094, 0.181] | 0.203 [0.084, 0.334] | 0.380 [0.328, 0.433] | 0.300 [0.257, 0.342] | 0.920 [0.862, 0.966] | 0.071 [0.024, 0.129] | 0.109 [0.103, 0.114] | 81 |
| Proportional | 0.425 [0.322, 0.529] | 0.000 [0.000, 0.000] | 0.163 [0.124, 0.205] | 0.086 [0.054, 0.123] | 0.243 [0.108, 0.378] | 0.274 [0.235, 0.315] | 0.305 [0.261, 0.350] | 0.862 [0.782, 0.931] | 0.129 [0.059, 0.200] | 0.106 [0.100, 0.111] | 83 |
| Random | 0.379 [0.276, 0.483] | 0.000 [0.000, 0.000] | 0.146 [0.110, 0.185] | 0.087 [0.055, 0.124] | 0.303 [0.152, 0.455] | 0.279 [0.236, 0.325] | 0.320 [0.281, 0.359] | 0.816 [0.736, 0.897] | 0.176 [0.094, 0.259] | 0.106 [0.100, 0.111] | 85 |

## Proportional minus the other method (paired)

| Metric | vs | Mean diff [95% CI] | Holm p |
|---|---|---|---|
| Has citation | Top-K | +0.046 [-0.080, +0.172] | 1 |
| Has citation | MMR | -0.011 [-0.161, +0.126] | 1 |
| Has citation | Random | +0.046 [-0.069, +0.161] | 1 |
| Invalid citation | Top-K | +0.000 [+0.000, +0.000] | 1 |
| Invalid citation | MMR | +0.000 [+0.000, +0.000] | 1 |
| Invalid citation | Random | +0.000 [+0.000, +0.000] | 1 |
| Faithful (NLI) | Top-K | +0.010 [-0.036, +0.056] | 1 |
| Faithful (NLI) | MMR | +0.001 [-0.045, +0.049] | 1 |
| Faithful (NLI) | Random | +0.017 [-0.032, +0.065] | 1 |
| Contradicted | Top-K | -0.027 [-0.063, +0.009] | 0.36 |
| Contradicted | MMR | -0.050 [-0.103, +0.001] | 0.231 |
| Contradicted | Random | -0.001 [-0.049, +0.045] | 0.974 |
| Citation precision | Top-K | +0.020 [-0.160, +0.200] | 1 |
| Citation precision | MMR | +0.000 [-0.222, +0.222] | 1 |
| Citation precision | Random | -0.174 [-0.435, +0.087] | 0.618 |
| Polarity gap to pool | Top-K | -0.047 [-0.104, +0.009] | 0.717 |
| Polarity gap to pool | MMR | -0.106 [-0.171, -0.040] | 0.0098 |
| Polarity gap to pool | Random | -0.005 [-0.055, +0.043] | 0.773 |
| Polarity gap to evidence | Top-K | +0.008 [-0.043, +0.058] | 1 |
| Polarity gap to evidence | MMR | +0.005 [-0.049, +0.059] | 1 |
| Polarity gap to evidence | Random | -0.015 [-0.069, +0.037] | 1 |
| Mentions a complaint | Top-K | +0.000 [-0.080, +0.080] | 1 |
| Mentions a complaint | MMR | -0.057 [-0.138, +0.011] | 0.395 |
| Mentions a complaint | Random | +0.046 [-0.046, +0.138] | 0.692 |
| Minority lost | Top-K | +0.000 [-0.082, +0.082] | 1 |
| Minority lost | MMR | +0.059 [-0.012, +0.141] | 0.395 |
| Minority lost | Random | -0.047 [-0.141, +0.047] | 0.692 |
| ROUGE-L | Top-K | +0.000 [-0.005, +0.005] | 1 |
| ROUGE-L | MMR | -0.003 [-0.009, +0.002] | 0.373 |
| ROUGE-L | Random | -0.000 [-0.006, +0.006] | 1 |
| Words | Top-K | -1.609 [-6.977, +3.724] | 1 |
| Words | MMR | +1.667 [-4.023, +7.276] | 1 |
| Words | Random | -1.828 [-7.977, +4.011] | 1 |
