RetNet

Retentive Network: A successor to Transformer for Large Language Models

https://arxiv.org/pdf/2307.08621.pdf

研究方向：

1. Transformer：Training Parallelism，Strong Performance
2. Linear Transformer：Training Parallelism，Low-Cost Inference
3. Strong Performance：Strong Performance，Training Parallelism
4. RetNet: Training Parallelism，Strong Performance，Low-Cost Inference

module：

1. Identical blocks 组成
2. multi-scale retention （MSR）module
3. Feed-Forward network （FFN）module

Retention 机制
$$
s_n = As_{n-1} + K^T_nv_n \\
on = Q_ns_n = \sum_{m=1}^nQ_nA^{n-m}K^T_mv_m
$$
模仿attention的QKV，$o_n$是后面推断需要用到的特征

