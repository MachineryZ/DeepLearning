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

代码部分：https://github.com/younglululu/DeepPINK/blob/master/run_withKnockoff_all.py
~~~python
import time
import math
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.linalg import qr

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, merge, LocallyConnected1D, Flatten, Conv1D
from keras import backend as K
from keras import regularizers
from keras.objectives import mse
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

dataDir = "/media/yanglu/TOSHIBA/data/featuresAndResponseDataframeCSVs/2018_6_15_fiveFoldCSVs

dataTypeList = ["LRnoMotifs"];
num_epochs = 200
batch_size = 10
filterNum = 1
bias = True
activation = "relu"
iterNum = 10

def calc_selectedfeat(origin_vec, knockoff_vec, q_thres):
    W = np.fabs(origin_vac) - np.fabs(knockoff_vec)
    print(W.shape)
    t = np.concatenate(([0], np.sort(np.fabs(W))))
    ratio = np.zeros(origin_vec)
    for j in range(origin_vec):
        ratio[j] = 1.0 * len(np.where(W <= -t[j][0]>) / np.max((1, len(np.where(W >= t[j])[0]))))
    T = np.inf
    arr = np.where(ratio <= q_thres)[0]
    if len(arr) > 0:
        id = np.min(arr)
        T = t[id]
    qualifiedIndices = np.where(np.fabs(W) >= T)[0];
    return qualifiedIndices

def show_layer_info(layer_name, layer_out):
    pass

def build_DNN(p, coeff=0):
    input = Input(name="input", shape=(p, 2))
    show_layer_info("Input", input)

    local1 = LocallyConnected1D(filterNum, 1, use_bias=bias, kernel_initializer=Constant(value=0.1))(input)
    local2 = LocallyConnected1D(1, 1, use_bias=bias, kernel_initializer="glorot_normal")(local1)
    flat = Flatten()(local2)
    dense1 = Dense(p, activation=activation, use_bias=bias, kernel_initializer="glorot_normal", kernel_regularizer=regularizers.l1(coeff))(flat)
    
~~~