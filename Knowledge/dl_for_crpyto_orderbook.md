# Deep Learning for Digital Asset Limit Order Books

Introduction: 本篇论文主要讨论的是对于数字货币的 lob 的建模讨论，digital asset 的 lob 往往会比正常的金融资产包含更加复杂的规律，本篇 paper 就是想要利用 deep learning 去预测 crpytocurrency limit order book 的 dynamics。之前的一些 machine learning 的方法一般是：bayesian neural networks，gradien boosting decision trees，lstm 等等。更甚者还有用 ARIMA 模型的。但是对于 cryptocurrency 的 high volatility 来说，risk management 是更加重要的。Sirignano 介绍了一种 spatial neural network 作为一种 low-dimensional means 以及、基于现在的 lob 的状态然后来预测未来的 lob 的状态 （CNN with limit order book data for stock price prediction）

<div align=center><img src="../Files/dl_for_crypto_orderbook1.jpg" width=90%></div>

Methodology: 
1. orderbook and mid price data:
    1. data: 由 100ms 的 snapshots 和 深度为 50 的 9个连续日，从2019.06.12 - 2019.06.20
    2. Features: 深度为 50 的 bid ask 的 price volume，总共 200 个 features
2. 数据处理
    1. price 和 volume 这种不同 asset 非常不一样的数据，我们首先需要归一化。$x_{norm} = \frac{x - \hat x}{\sigma_x}$ 
3. 价格变动：
    1. 主要的价格变动都是由前 10 档的 ask 和 bid 造成的。所以对于后 40 档的数据，我们没有必要增加这么多计算量而只得到一些 marginal 的 提升
    

