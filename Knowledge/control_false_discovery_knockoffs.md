# CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS

https://arxiv.org/pdf/1404.5609.pdf

这篇 paper 主要介绍了 knockoof filter，一种新型的变量选择过程，可以控制 FDR 在 linear model 里的特征选择。对于传统的线性模型来说：
$$
y = X \beta + z
$$
其中，$y\in R^n, X\in R^{n\times p},\beta \in R^p$，我们考虑 $n\geq p$ 的情况