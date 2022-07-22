# Temporal Concatenation for Markov Decision Process

TC
https://arxiv.org/pdf/2004.11555.pdf

这篇 paper 主要描述的是如何将一个在 mdp 中探索的一个 episode 运用分治算法得到子问题的子最优解之后
再将这些若干子最优价拼接起来，得到一个近似全局最优解的过程。这个算法我们成为 temporal concatenation 
算法（TC）。TC 算法在之后的论证中我们可以看到有诸多好处，首先，每个子问题的解过程可以并行，也即，近似解的时间
复杂度会比原始问题解的时间复杂度少非常多。

