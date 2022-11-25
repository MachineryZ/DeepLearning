# Deep Learning for Digital Asset Limit Order Books

Introduction: 本篇论文主要讨论的是对于数字货币的 lob 的建模讨论，digital asset 的 lob 往往会比正常的金融资产包含更加复杂的规律，本篇 paper 就是想要利用 deep learning 去预测 crpytocurrency limit order book 的 dynamics。之前的一些 machine learning 的方法一般是：bayesian neural networks，gradien boosting decision trees，lstm 等等。更甚者还有用 ARIMA 模型的。但是对于 cryptocurrency 的 high volatility 来说，risk management 是更加重要的。Sirignano 介绍了一种 spatial neural network 作为一种 low-dimensional means 以及、基于现在的 lob 的状态然后来预测未来的 lob 的状态 （CNN with limit order book data for stock price prediction）


