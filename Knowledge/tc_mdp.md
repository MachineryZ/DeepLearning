# Temporal Concatenation for Markov Decision Process

TC
https://arxiv.org/pdf/2004.11555.pdf

这篇 paper 主要描述的是如何将一个在 mdp 中探索的一个 episode 运用分治算法得到子问题的子最优解之后
再将这些若干子最优价拼接起来，得到一个近似全局最优解的过程。这个算法我们成为 temporal concatenation 
算法（TC）。TC 算法在之后的论证中我们可以看到有诸多好处，首先，每个子问题的解过程可以并行，也即，近似解的时间
复杂度会比原始问题解的时间复杂度少非常多。

system set-up

有限时间 horizon 是 $[T]$，state space $\mathcal{S}$，action set $\mathcal{A}$，对于系统里的转换方程
$$
S_{t+1} = p_t(a_t, S_t, Y_t^S), \ \ t\in[T]
$$
其中 $Y_t^S$ 是一个随机变量，用来描述 mdp 的状态转移过程概率。同样的我们也可以写出 reward 和 action 的表达形式：
$$
R_t = R_t(a_t, S_t, Y_t^R) \\
a_t=\pi(t, S_t, Y^P)
$$
假设 reward 能被一个 upper bound 来限制住，称为 $\bar r$，假设，一个 mdp 算法，称为一个 ALG，那么假设对于集合 $\mathcal{I}_0$ 由 alg 解出的最优策略是：
$$
\pi^* = ALG(\mathcal{I}_0)
$$

那么，我们可以得到 temporal concatenation 的定义：
$$
\mathcal{I_1}\triangleq(R_{0,T/2-1}, p_{0,T/2-1}, T/2) \\
\mathcal{I_2}\triangleq(R_{T/2,T-1}, p_{T/2,T-1}, T/2) \\
\pi_{1}^*\triangleq ALG(\mathcal I_1) \\
\pi_{2}^*\triangleq ALG(\mathcal I_2)
$$
那么我们就可以直接做一个拼接，得到 TC 的策略 $\pi_{TC}$，我们要考察的就是 TC 策略和真实的最优策略的 error 是多少：
$$
\Delta(\mathcal I_0,\mu_0) \triangleq V(\mathcal I_0, \pi^*, \mu_0) - V(\mathcal I_0,\pi_{TC}, \mu_0)
$$
通过一些数学推导和证明，以及对于 mdp 子问题的更加严格的 diameter 的定义之后，我们可以得到一个 upper bound
$$
\Delta(\mathcal I_0, \mu_0) \leq \bar r \tau_\epsilon (\mathcal I_0) / (1 - \epsilon)
$$
以及 lower bound
$$
\Delta(\mathcal I_0,\mu_0) = (\tau_0(\mathcal I_0) - 2) \bar r - \sigma
$$
我们也发现，这个 upper bound 和 lower bound 相差的仅仅是一个常数，说明这个bound还是比较紧的

实验中，本文使用了一个 deterministic graph traversal 的图结构 mdp 来验证我们 tc 的算法复杂度、error 在划分不同子问题的情况下的表现。


