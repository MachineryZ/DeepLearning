# DeepPINK
DeepPINK: reproducible feature selection in deep neural networks
https://proceedings.neurips.cc/paper/2018/file/29daf9442f3c0b60642b14c081b4a556-Paper.pdf

FDR false discovery rate 这个技术，貌似在 cnotrol for fdr 这篇 paper 中提到过，细节可以看这篇 paper。

model setting，假设我们是监督学习的任务，那么我们有若干 i.i.d. 的配对 $(x_i, Y_i)$，其中 x 是向量，y是标量，假设有一个子集 $S_0\subset$ \{\1,...,p}$，那么我们的目标是，找到这个子集 $S_0$ 使得他的回归结果 $Y_i$，是和补集 $S_0^c$ 是无关的。

FDR control and knockoff filter：对于一个选择出来的因子集合 $\hat S$，他的 FDR 会定义为：
$$
FDR = E[FDP] \text{with } FDP = \frac{|\hat S\cap S_0^c|}{|\hat S|}
$$

对于之前的 paper 来说，fdr 这个 tech 其实并没办法很好的适配到 deep learing 上面。所以，本篇工作，主要 focus 在 model-X knockoffs framework （https://arxiv.org/abs/1610.02351.pdf）

定义1：Model-X knockoff features 是对于一簇随机因子 $x = (X_1, ..., X_p)^T$ 的一个新簇 $\tilde{x} = \{\tilde X_1, ..., \tilde X_p\}$，使得这个新簇满足两条性质:
1. $(x, \tilde x)_{swap(S)}  \triangleq (x, \tilde x)$ 对于任意 $S\subset \{1,2,...,p\}$，其中 $swap(S)$ 意思是 交换 $X_j$ 和 $X_i$ 对于任意 $S$ 中的两个元素，然后 $\triangleq$ 定义为同分布
2. $\tilde x\perp Y|x$，表示 $\tilde x$ 是独立于 Y given feature x

构造 knockoff features 的 分布：
$$
\tilde x | x \sim N(x - diag\{s\}\Sigma^{-1}x, 2diag\{s\}-siga\{s\}\Sigma^{-1}diag\{s\})
$$
其中 diag{s} 是对角矩阵上元素都为 s 的方阵。所以，对于 model-X knockoff features 会有以下的 joint distribution
$$

$$