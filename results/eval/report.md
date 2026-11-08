# SPACE selection, statistical evaluation

100 hotel-aspect cases (50 hotels x cleanliness and service), up to 20 candidates, k = 5. Random is the mean of 500 random draws per case. Numbers is mean with 95% bootstrap CI (10000 resamples over the cases).

Our own implementations in src/eval/core.py picks the same reviews as src/space/ in: Top-K 100/100, MMR 100/100, Proportional 100/100.

## Aspect: all

| Method | Sentiment JS distance | TVD over 1-5 stars | Minority lost | Positive bias (sel - pool) | Relevance | Semantic diversity |
|---|---|---|---|---|---|---|
| Top-K | 0.249 [0.225, 0.273] | 0.280 [0.257, 0.305] | 0.173 [0.102, 0.255] | -0.091 [-0.130, -0.054] | 0.539 [0.532, 0.546] | 0.400 [0.391, 0.410] |
| MMR | 0.253 [0.228, 0.279] | 0.299 [0.275, 0.324] | 0.092 [0.041, 0.153] | -0.137 [-0.173, -0.102] | 0.514 [0.507, 0.520] | 0.490 [0.481, 0.498] |
| Proportional | 0.230 [0.209, 0.252] | 0.251 [0.229, 0.273] | 0.235 [0.153, 0.327] | 0.007 [-0.025, 0.037] | 0.529 [0.521, 0.536] | 0.354 [0.346, 0.362] |
| Random | 0.232 [0.222, 0.242] | 0.261 [0.252, 0.270] | 0.222 [0.180, 0.267] | 0.001 [-0.000, 0.003] | 0.479 [0.473, 0.484] | 0.409 [0.403, 0.415] |
| Oracle (best JS) | 0.113 [0.101, 0.124] | - | - | - | - | - |

## Aspect: cleanliness

| Method | Sentiment JS distance | TVD over 1-5 stars | Minority lost | Positive bias (sel - pool) | Relevance | Semantic diversity |
|---|---|---|---|---|---|---|
| Top-K | 0.273 [0.239, 0.309] | 0.310 [0.277, 0.344] | 0.122 [0.041, 0.224] | -0.119 [-0.179, -0.058] | 0.556 [0.546, 0.566] | 0.391 [0.379, 0.404] |
| MMR | 0.286 [0.252, 0.322] | 0.332 [0.302, 0.362] | 0.041 [0.000, 0.102] | -0.175 [-0.228, -0.121] | 0.527 [0.518, 0.536] | 0.486 [0.473, 0.498] |
| Proportional | 0.242 [0.211, 0.273] | 0.263 [0.234, 0.292] | 0.204 [0.102, 0.327] | -0.007 [-0.056, 0.042] | 0.546 [0.535, 0.557] | 0.353 [0.343, 0.362] |
| Random | 0.238 [0.223, 0.251] | 0.270 [0.259, 0.281] | 0.172 [0.121, 0.230] | 0.002 [-0.000, 0.004] | 0.486 [0.478, 0.494] | 0.410 [0.402, 0.417] |
| Oracle (best JS) | 0.109 [0.093, 0.124] | - | - | - | - | - |

## Aspect: service

| Method | Sentiment JS distance | TVD over 1-5 stars | Minority lost | Positive bias (sel - pool) | Relevance | Semantic diversity |
|---|---|---|---|---|---|---|
| Top-K | 0.225 [0.193, 0.257] | 0.250 [0.218, 0.283] | 0.224 [0.122, 0.347] | -0.064 [-0.110, -0.019] | 0.523 [0.515, 0.531] | 0.409 [0.395, 0.423] |
| MMR | 0.220 [0.187, 0.254] | 0.266 [0.230, 0.302] | 0.143 [0.061, 0.245] | -0.100 [-0.144, -0.055] | 0.501 [0.493, 0.508] | 0.494 [0.482, 0.505] |
| Proportional | 0.219 [0.190, 0.247] | 0.238 [0.206, 0.271] | 0.265 [0.143, 0.388] | 0.020 [-0.018, 0.058] | 0.512 [0.503, 0.520] | 0.356 [0.343, 0.369] |
| Random | 0.226 [0.210, 0.240] | 0.252 [0.239, 0.265] | 0.271 [0.209, 0.336] | 0.000 [-0.002, 0.002] | 0.471 [0.464, 0.478] | 0.408 [0.400, 0.416] |
| Oracle (best JS) | 0.116 [0.099, 0.133] | - | - | - | - | - |

## Paired differences (a - b), all cases

| Metric | a vs b | Mean diff [95% CI] | Wilcoxon p | Holm p |
|---|---|---|---|---|
| Sentiment JS distance | Proportional vs Top-K | -0.019 [-0.044, +0.004] | 0.173 | 0.932 |
| Sentiment JS distance | Proportional vs MMR | -0.023 [-0.053, +0.006] | 0.155 | 0.932 |
| Sentiment JS distance | Proportional vs Random | -0.002 [-0.020, +0.017] | 0.606 | 1 |
| Sentiment JS distance | Top-K vs Random | +0.017 [-0.004, +0.038] | 0.43 | 1 |
| Sentiment JS distance | MMR vs Random | +0.021 [-0.001, +0.043] | 0.221 | 0.932 |
| Sentiment JS distance | MMR vs Top-K | +0.004 [-0.023, +0.032] | 0.745 | 1 |
| TVD over 1-5 stars | Proportional vs Top-K | -0.030 [-0.055, -0.006] | 0.028 | 0.112 |
| TVD over 1-5 stars | Proportional vs MMR | -0.048 [-0.076, -0.021] | 0.00196 | 0.0118 |
| TVD over 1-5 stars | Proportional vs Random | -0.011 [-0.029, +0.009] | 0.205 | 0.614 |
| TVD over 1-5 stars | Top-K vs Random | +0.019 [-0.004, +0.042] | 0.242 | 0.614 |
| TVD over 1-5 stars | MMR vs Random | +0.038 [+0.016, +0.060] | 0.00518 | 0.0259 |
| TVD over 1-5 stars | MMR vs Top-K | +0.019 [-0.010, +0.048] | 0.238 | 0.614 |
| Minority lost | Proportional vs Top-K | +0.061 [-0.010, +0.133] | 0.0833 | 0.124 |
| Minority lost | Proportional vs MMR | +0.143 [+0.051, +0.235] | 0.00427 | 0.0171 |
| Minority lost | Proportional vs Random | +0.013 [-0.057, +0.086] | 0.0618 | 0.124 |
| Minority lost | Top-K vs Random | -0.048 [-0.110, +0.014] | 0.00041 | 0.00205 |
| Minority lost | MMR vs Random | -0.130 [-0.181, -0.078] | 2.6e-09 | 1.56e-08 |
| Minority lost | MMR vs Top-K | -0.082 [-0.153, -0.010] | 0.0325 | 0.0975 |
| Positive bias (sel - pool) | Proportional vs Top-K | +0.098 [+0.070, +0.128] | 2.48e-08 | 9.92e-08 |
| Positive bias (sel - pool) | Proportional vs MMR | +0.144 [+0.102, +0.186] | 6.13e-09 | 3.06e-08 |
| Positive bias (sel - pool) | Proportional vs Random | +0.005 [-0.025, +0.036] | 0.861 | 0.861 |
| Positive bias (sel - pool) | Top-K vs Random | -0.093 [-0.130, -0.054] | 1.03e-05 | 3.08e-05 |
| Positive bias (sel - pool) | MMR vs Random | -0.139 [-0.174, -0.103] | 2.9e-10 | 1.74e-09 |
| Positive bias (sel - pool) | MMR vs Top-K | -0.046 [-0.082, -0.010] | 0.0321 | 0.0643 |
| Relevance | Proportional vs Top-K | -0.011 [-0.012, -0.009] | 2.6e-17 | 5.21e-17 |
| Relevance | Proportional vs MMR | +0.015 [+0.012, +0.019] | 4.26e-12 | 4.26e-12 |
| Relevance | Proportional vs Random | +0.050 [+0.046, +0.054] | 3.9e-18 | 2.34e-17 |
| Relevance | Top-K vs Random | +0.061 [+0.057, +0.064] | 3.9e-18 | 2.34e-17 |
| Relevance | MMR vs Random | +0.035 [+0.032, +0.038] | 3.9e-18 | 2.34e-17 |
| Relevance | MMR vs Top-K | -0.026 [-0.028, -0.023] | 3.9e-18 | 2.34e-17 |
| Semantic diversity | Proportional vs Top-K | -0.046 [-0.054, -0.038] | 4.27e-15 | 8.53e-15 |
| Semantic diversity | Proportional vs MMR | -0.135 [-0.145, -0.126] | 3.9e-18 | 2.34e-17 |
| Semantic diversity | Proportional vs Random | -0.055 [-0.061, -0.049] | 1.33e-17 | 4e-17 |
| Semantic diversity | Top-K vs Random | -0.009 [-0.018, +0.001] | 0.0605 | 0.0605 |
| Semantic diversity | MMR vs Random | +0.081 [+0.074, +0.087] | 3.9e-18 | 2.34e-17 |
| Semantic diversity | MMR vs Top-K | +0.090 [+0.080, +0.100] | 4.27e-18 | 2.34e-17 |

## Ablation of the proportional score (mean over all cases)

penalty 2 = p_S - 2p_R (current code, exact HPF gain), 1 = p_S - p_R, 0 = only p_S. first: hpf = current code, relevance = first pick on relevance only like IAdU (Eq. 17).

| lambda | penalty | first pick | JS | minority lost | pos. bias | relevance | diversity |
|---|---|---|---|---|---|---|---|
| 0.0 | 0 | hpf | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.0 | 0 | relevance | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.0 | 1 | hpf | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.0 | 1 | relevance | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.0 | 2 | hpf | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.0 | 2 | relevance | 0.249 | 0.173 | -0.091 | 0.539 | 0.400 |
| 0.25 | 0 | hpf | 0.247 | 0.245 | -0.017 | 0.536 | 0.363 |
| 0.25 | 0 | relevance | 0.247 | 0.245 | -0.017 | 0.536 | 0.363 |
| 0.25 | 1 | hpf | 0.243 | 0.245 | -0.023 | 0.537 | 0.370 |
| 0.25 | 1 | relevance | 0.243 | 0.245 | -0.023 | 0.537 | 0.370 |
| 0.25 | 2 | hpf | 0.245 | 0.255 | -0.033 | 0.537 | 0.379 |
| 0.25 | 2 | relevance | 0.245 | 0.255 | -0.033 | 0.537 | 0.379 |
| 0.5 | 0 | hpf | 0.240 | 0.255 | +0.029 | 0.520 | 0.323 |
| 0.5 | 0 | relevance | 0.233 | 0.245 | +0.021 | 0.524 | 0.337 |
| 0.5 | 1 | hpf | 0.235 | 0.245 | +0.015 | 0.524 | 0.338 |
| 0.5 | 1 | relevance | 0.233 | 0.235 | +0.007 | 0.526 | 0.346 |
| 0.5 | 2 | hpf | 0.230 | 0.235 | +0.007 | 0.529 | 0.354 |
| 0.5 | 2 | relevance | 0.232 | 0.235 | +0.003 | 0.530 | 0.358 |
| 0.75 | 0 | hpf | 0.247 | 0.347 | +0.093 | 0.495 | 0.291 |
| 0.75 | 0 | relevance | 0.227 | 0.276 | +0.063 | 0.507 | 0.317 |
| 0.75 | 1 | hpf | 0.235 | 0.316 | +0.075 | 0.502 | 0.305 |
| 0.75 | 1 | relevance | 0.223 | 0.265 | +0.053 | 0.511 | 0.328 |
| 0.75 | 2 | hpf | 0.228 | 0.286 | +0.053 | 0.508 | 0.325 |
| 0.75 | 2 | relevance | 0.220 | 0.255 | +0.041 | 0.514 | 0.341 |
| 1.0 | 0 | hpf | 0.258 | 0.439 | +0.121 | 0.481 | 0.282 |
| 1.0 | 0 | relevance | 0.228 | 0.306 | +0.079 | 0.497 | 0.313 |
| 1.0 | 1 | hpf | 0.248 | 0.388 | +0.107 | 0.482 | 0.294 |
| 1.0 | 1 | relevance | 0.223 | 0.265 | +0.073 | 0.497 | 0.321 |
| 1.0 | 2 | hpf | 0.242 | 0.357 | +0.103 | 0.485 | 0.311 |
| 1.0 | 2 | relevance | 0.229 | 0.296 | +0.081 | 0.498 | 0.334 |
