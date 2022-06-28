# VAE

Auto-Encoding Variantional Bayes

直觉：对于一个 high dimensional 的 random variable $x$，我们想要从某一个条件概率分布来生成 $p_\theta(x|z)$。通常来说 z 的维度会比 x 的维度少很多。那么，得到 z 也需要一个分布 $p_\theta(z)$。所以完整的生成模型表达式应该是 $p_\theta(z) p_\theta(x|z)$

1. 想根据 $x$ 得到 $z$
    1. $p_\theta(x) = p_\theta(x|z)p_\theta(z)/p_\theta(x)$
2. 想得到 $x$ 的分布估计
    1. $p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$
3. 使用一个神经网络 $q_\phi$ 来拟合
    1. $p_\theta(z|x) = q_\phi(z|x)$

那么我们这里就设置好了网络的输出，loss function 就用 kl divergence 来设计：
1. $D_{KL}(q_\phi(z|x)||p_\theta(z|x))=-\sum_z q_\phi (z|x)[\log(\frac{p_\theta(x,z)}{q_\phi(z|x)}-\log(p_\theta(x)))]$
2. $\log(p_\theta(x))=D_{KL}(q_\phi(z|x)||p_\theta(z|x))+L(\theta,\phi; x)$（variation lower bound）


通俗的解释就是：像 autoencoder 类的模型，在 hidden space 空间里，都是离散的点，来进行映射（因为数据有限，所以只能覆盖空间中有限点）那么，为了覆盖整个空间，我们会加上噪声，但是噪声范围又有限，所以加上了无限范围的高斯噪声。
