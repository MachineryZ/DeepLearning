# Universal Trading for Order Execution with Oracle Policy Distillation

https://arxiv.org/pdf/2103.10860v1.pdf

这篇 paper 的主要贡献有：
1. learning-based + model-free 的方法不仅可以比传统方法要好，也可以得到一个可解释的、有利润的交易执行策略
2. teacher-student learning paradigm 对于 policy distillation 可以减轻 imperfect information 与 optimal decision making 的 gap
3. 这是第一篇发掘不同的 instruments 然后开发出一个普适的最优交易策略

Methodology
1. Formulation of Order Execution
    1. Discrete Timesteps, $\{0,1,...,T-1\}$ with respective price $\{p_0, p_1, ..., p_{T-1}\}$
    2. At each timestep, the trader will propose to trade a volume of $q_{t+1} \geq 0$ shares
    3. 这个 trading order 会在 price $p_{t+1}$ 被交易（也是 market price）
    4. 我们的目标是最大化利润（市场完全流动）



