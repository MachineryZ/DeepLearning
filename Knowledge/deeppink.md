# DeepPINK
DeepPINK: reproducible feature selection in deep neural networks
https://proceedings.neurips.cc/paper/2018/file/29daf9442f3c0b60642b14c081b4a556-Paper.pdf

FDR false discovery rate 这个技术，貌似在 cnotrol for fdr 这篇 paper 中提到过，细节可以看这篇 paper。

model setting，假设我们是监督学习的任务，那么我们有若干 i.i.d. 的配对 $(x_i, Y_i)$，其中 x 是向量，y是标量，假设有一个子集 $S_0\subset$ \{\1,...,p}$，那么我们的目标是，找到这个子集 $S_0$ 使得他的回归结果 $Y_i$，是和补集 $S_0^c$ 是无关的。

FDR control and knockoff filter：对于一个选择出来的因子集合 $\hat S$，他的 FDR 会定义为：
$$
FDR = E[FDP] \text{with } FDP = \frac{|\hat S\cap S_0^c|}{|\hat S|}
$$

对于之前的 paper 来说，fdr 这个 tech 其实并没办法很好的适配到 deep learing 上面。