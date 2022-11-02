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
    4. 我们的目标是最大化利润（市场完全流动）: $argmax_{q_1,q_2,...,q_T}\sum_{t=0}^{T-1}(q_{t+1} p_{t+1},s.t.\sum_{t=0}^{T-1}q_{t+1}=Q$
    5. 平均执行价格 average execution price AEP 是 $\sum_{t=0}^{T-1}q_{t+1}p_{t+1}/Q$
    6. 也就是说，为了得到更多的利润，我们需要更合理的分配 $q_{t+1}$ 使得平均价格更高
    7. 
2. Order Execution as a Markov Decision Process
    1. State：observed state $s_t$ at the timestep t 是描述整个市场和交易者的信息
        1. private variable of trader 和 public variable of market
        2. private variable 包括时间 t 还有 剩余的 inventory $Q - \sum_{i=1}^t q_i$ 可以交易的量
        3. public variable 是 open，high，low，close，average price 和 transaction volume of each time step
    2. Action
        1. 对于某一个时刻 t，$a_t$ 代表着，下一个 t+1 时刻交易的 volume 则是 $q_{t+1} = a_t Q$，action 会被 standardized trading volume
        2. $\sum_{t=0}^{T-1} a_t = 1.0$
        3. $a_{T-1} = max\{1 - \sum_{i=0}^{T-2} a_i, \pi(s_{T-1})$
        4. 但是，我们不希望 $a_t$ 的分布有极端值，所以会让交易尽可能地平滑
    3. Reward
        1. Reward 部分分为两个，第一个是 trading profitability，也就是交易收益；第二个是 market impact penalty，也就是市场影响惩罚
        2. 为了描述 trading profitability，我们会将 positive part of reward 定义为 volume weighted price advantage $\hat R^{+}_t(s_t,a_t) = \frac{q_{t+1}}{Q} (\frac{p_{t+1} - \bar p }{\bar p})=a_t(\frac{p_{t+1} - \bar p }{\bar p})$，其中 $\bar p = \sum_{i=0}^{T-1}p_{i+1}/T$ 是averaged original market price （也就是，每一笔交易当前的收益率的）
        3. 对于当前交易可能会对市场有所冲击的情况，我们会用 quadratic penalty $\hat R^-_t=-\alpha (a_t)^2$
        4. 最终的 reward 表达式为 $R_t(s_t, a_t)=\hat R_t^+(s_t,a_t) + \hat R^-_t(s_t, a_t)= (\frac{p_{t+1}}{\bar p}- 1) a_t - \alpha (a_t)^2$
        5. 最终的 episode reward 则是用 discounted reward形式即可
3. Policy Distillation
    1. policy distillation 的本质是从不完全的市场信息中得到最优的交易规则，以及 rl 并不能很好的在噪音市场中获取一个好的策略
    2. 所以为了保证 sample 的有效性，我们使用 two stage 的 teacher-student policy distillation
    3. teacher: 是为了获得 optimal trading policy 的 nn，可以从环境获取完美、无噪音的信息
    4. student：是为了在有噪音的市场中，高效的学出来一个策略，然后从 teacher 网络与有噪音的市场中互动，distill 出来的网络
    5. 在有噪音市场中和 optimal trading policy 之间，建立一个联系，就是 policy distillation loss $L_d = -E_t[log Pr(a_t=\bar a_t|\pi_\theta, s_t;\pi_\phi, \bar s_t)]$，这个 loss 看起来就是在两种市场情况下，用不同的策略他们的期望大小是否一样的一个loss
4. Policy Optimization
    1. 在优化算法里，我们有如下几个 loss
    2. $L_p(\theta) = -E_t[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_old}(a_t|s_t)}\hat A(s_t, a_t) - \beta KL(\pi\theta_{old}(\cdot|s_t), \pi_\theta(\cdot|s_t))]$，这是最小化 objective function 的loss，
    3. $L_v(\theta) = E_t[||V_\theta(s_t) - V_t||_2]$  
    4. $L(\theta) = L_p + \lambda L_v + \mu L_d$
    

<div align=center><img src="../Files/policydistillation.jpg" width=70%></div>


market impact
participation rate: pp * volatility(sd) + spread
理解：spread 变得比较大之后，可以认为市场冲击的比较大。