# Factorization Machines Model
因子分解机模型，factorization machines model，简称 FM

Cholesky 分解，设 $\mathcal{A}$ 是 n 阶对称正定矩阵，则存在唯一的对角元均为正数的下三角矩阵 $\mathcal{L}$，使得 $\mathcal{A} = \mathcal{LL^T}$

线性回归模型假设特征之间是相互独立的、不相关的。但在现实的某些场景中，特征之间往往是相关的，而不是相互独立的（尤其是在推荐系统中），所以很多场景需要特征组合。如何组合？假设有 n 个特征，分别是 $x_1, x_2, ..., x_n$，则两两组合，可以构成一个 $n \times n$ 的对称矩阵

二姐多项式回归模型（Polynomial Regression）方程如下：$y = w_0 + \sum_{i=1}^{n}w_i x_i + \sum_i^n\sum_{j\geq i}^n w_{ij}x_ix_j$，其中 $w_0, w_i, w_{ij}$ 是模型参数。但二姐多项式回归模型的局限在于：样本中没有出现交互的特征组合，不能对相应的参数进行估计。

特征关系的向量化，则用到我们的 FM 模型：二项式参数可以组成一个对称矩阵 $\mathcal{W}$，根据 Cholesky 分解，则可以分解为 $\mathcal{W} = \mathcal{VV^T}$，其中 $\mathcal{V}$ 的第 j 列便是第 j 维特征的隐向量。即每个参数 $w_{ij} = <\mathcal{V_i}, \mathcal{V_j}>$，最终可以改写成: $y = w_0 + \sum_{i=1}^{n}w_i x_i + \sum_i^n\sum_{j\geq i}^n <\mathcal{v_i}, \mathcal{v_j}>x_ix_j$

因为有 n 个 $v_i$ 来表示大矩阵，我们在实际中计算的时候，可以将时间复杂度为 $O(kn^2)$ 等价转化为 $O(kn)$
$y = w_0 + \sum_{i=1}^n w_i x_i + \frac{1}{2}\sum_{f=1}^k[(\sum_{i=1}^{n}v_{i,f}x_i)^2 - \sum_{i=1}^nv_{i,f}^2x_i^2]$，首先大大降低了复杂度、其次如果是稀疏特征，也可以不用 dense 的向量化计算来得到最后的结果

损失函数：FM 模型可用于回归（Regression），二分类（Binary classification），排名（Ranking）任务。回归任务对应最小平方误差、二分类任务用对数损失函数和交叉熵损失函数、排序任务用成对分类函数。训练方法有三种：随机梯度下降法（Stochastic Gradient Descent，SGD），交替最小二乘法（Alternating Least-Squares，ALS），马尔可夫蒙特卡洛法（Markov Chain Mopnte Carlo，MCMC）





