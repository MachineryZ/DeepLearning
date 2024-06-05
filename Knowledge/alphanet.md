个股量价 (number_of_stats, time) 是输入维度

|             |      |      |      |      |           |
| ----------- | ---- | ---- | ---- | ---- | --------- |
| open(t-n)   |      |      | ...  | ...  | open(t)   |
| high(t-n)   |      |      | ...  | ...  | high(t)   |
| low(t-n)    |      |      | ...  | ...  | low(t)    |
| close(t-n)  |      |      | ...  | ...  | close(t)  |
| vwap(t-n)   |      |      | ...  | ...  | vwap(t)   |
| volume(t-n) |      |      | ...  | ...  | volume(t) |
| return(t-n) |      |      | ...  | ...  | turn(t)   |
| turn(t-n)   |      |      |      |      |           |
|             |      |      |      |      |           |

值得注意的是，我们可以用行排列也可以用列排列；target y 应该就是。$R_{t+m} $ 也就是对应时间范围内的收益率

自定义的网络层有这些：

ts_cor ts_cov ts_stddev ts_zscore ts_return ts_decaylinear

2目运算

其中，如果是2目计算，则会计算$C_N^2$次，然后flatten展开输入到 Linear 层中

1目运算

和普通卷机类似，并无异常

BN

BN能缓解 internal covariate shift 使得学习更加稳定 加速模型训练

----

AlphaNet 模型构建和测试细节

~~~python
class Inception(nn.Module):
    """
    Inception, 用于提取时间序列的特征，具体操作包括：

    1. kernel_size 和 stride 均为 d=10 的特征提取层，类似于卷积层，用于提取时间序列的特征。具体包括：
        1. ts_corr4d: 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
        2. ts_cov4d: 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
        3. ts_stddev4d: 过去 d 天 X 值构成的时序数列的标准差
        4. ts_zscore4d: 过去 d 天 X 值构成的时序数列的平均值除以标准差
        5. ts_return4d: (X - delay(X, d))/delay(X, d)-1, 其中 delay(X, d) 为 X 在 d 天前的取值
        6. ts_decaylinear4d: 过去 d 天 X 值构成的时序数列的加权平均值，权数为 d, d – 1, …, 1（权数之和应为 1，需进行归一化处理），其中离现在越近的日子权数越大
        7. ts_mean4d: 过去 d 天 X 值构成的时序数列的平均值

        各操作得到的张量维数：
        1. 由于涉及两个变量的协方差，因此 ts_corr4d 和 ts_cov4d 的输出为 N*1*36*3
        2. 其余操作均只涉及单变量的时序计算，因此输出为 N*1*9*3

    2. 对第 1 步的输出进行 Batch Normalization 操作，输出维数仍为 N*1*36*3 或 N*1*9*3

    3. 对于第 2 步得到的张量，kernel_size 为 3 的池化层。具体包括：
        1. max_pool: 过去 d 天 X 值构成的时序数列的最大值
        2. avg_pool: 过去 d 天 X 值构成的时序数列的平均值
        3. min_pool: 过去 d 天 X 值构成的时序数列的最小值

        以上三个操作的输出均为 N*1*117*1

    4. 对第 3 步的输出进行 Batch Normalization 操作，输出维数仍为 N*1*117*1

    5. 将第 2 步和第 4 步的输出展平后进行拼接，得到的张量维数为 N*(2*36*3+5*9*3+3*117) = N*702

    """

    def __init__(self, combination, combination_rev, index_list):
        """
        combination: 卷积操作时需要的两列数据的组合
        combination_rev: 卷积操作时需要的两列数据的组合，与 combination 相反
        index_list: 卷积操作时需要的时间索引

        """

        super(Inception, self).__init__()
        # 卷积操作时需要的两列数据的组合
        self.combination = combination
        self.combination_rev = combination_rev

        # 卷积操作时需要的时间索引
        self.index_list = index_list
        self.d = len(index_list) - 1

        # 卷积操作后的 Batch Normalization 层
        self.bc1 = nn.BatchNorm2d(1)
        self.bc2 = nn.BatchNorm2d(1)
        self.bc3 = nn.BatchNorm2d(1)
        self.bc4 = nn.BatchNorm2d(1)
        self.bc5 = nn.BatchNorm2d(1)
        self.bc6 = nn.BatchNorm2d(1)
        self.bc7 = nn.BatchNorm2d(1)

        # 池化层，尺度为 1*d
        self.max_pool = nn.MaxPool2d(kernel_size=(1, self.d))
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, self.d))
        # 最小池化等价于相反数的最大池化，后续会对结果取反
        self.min_pool = nn.MaxPool2d(kernel_size=(1, self.d))

        # 池化操作后的 Batch Normalization 层
        self.bc_pool1 = nn.BatchNorm2d(1)
        self.bc_pool2 = nn.BatchNorm2d(1)
        self.bc_pool3 = nn.BatchNorm2d(1)

    def forward(self, data):
        """
        data: 输入的数据，维度为 batch_size*1*9*30

        """
        # 本层的输入为 batch_size*1*9*30, 在训练时不需要反向传播，因此可以使用 detach() 函数
        data = data.detach().cpu().numpy()
        combination = self.combination
        combination_rev = self.combination_rev

        # 卷积操作
        conv1 = self.ts_corr4d(data, combination, combination_rev).to(torch.float)
        conv2 = self.ts_cov4d(data, combination, combination_rev).to(torch.float)
        conv3 = self.ts_stddev4d(data).to(torch.float)
        conv4 = self.ts_zcore4d(data).to(torch.float)
        conv5 = self.ts_return4d(data).to(torch.float)
        conv6 = self.ts_decaylinear4d(data).to(torch.float)
        conv7 = self.ts_mean4d(data).to(torch.float)

        # 卷积操作后的 Batch Normalization
        batch1 = self.bc1(conv1)
        batch2 = self.bc2(conv2)
        batch3 = self.bc3(conv3)
        batch4 = self.bc4(conv4)
        batch5 = self.bc5(conv5)
        batch6 = self.bc6(conv6)
        batch7 = self.bc7(conv7)

        # 在 H 维度上进行特征拼接
        feature = torch.cat(
            [batch1, batch2, batch3, batch4, batch5, batch6, batch7], axis=2
        )  # N*1*(2*36+5*9)*3 = N*1*117*3

        # 同时将特征展平，准备输入到全连接层
        feature_flatten = feature.flatten(start_dim=1)  # N*(117*3) = N*351

        # 对多通道特征进行池化操作，每层池化后面都有 Batch Normalization
        # 最大池化
        maxpool = self.max_pool(feature)  # N*1*117*1
        maxpool = self.bc_pool1(maxpool)
        # 平均池化
        avgpool = self.avg_pool(feature)  # N*1*117*1
        avgpool = self.bc_pool2(avgpool)
        # 最小池化
        # N*1*117*1, 最小池化等价于相反数的最大池化，并对结果取反
        minpool = -self.min_pool(-1 * feature)
        minpool = self.bc_pool3(minpool)
        # 特征拼接
        pool_cat = torch.cat(
            [maxpool, avgpool, minpool], axis=2
        )  # N*1*(3*117)*1 = N*1*351*1
        # 将池化层的特征展平
        pool_cat_flatten = pool_cat.flatten(start_dim=1)  # N*351

        # 拼接展平后的特征
        feature_final = torch.cat(
            [feature_flatten, pool_cat_flatten], axis=1
        )  # N*(351+351) = N*702
        return feature_final

    # 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
    def ts_corr4d(self, Matrix, combination, combination_rev):
        ...

    # 过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
    def ts_cov4d(self, Matrix, combination, combination_rev):
        ...

    # 过去 d 天 X 值构成的时序数列的标准差
    def ts_stddev4d(self, Matrix):
        ...

    # 过去 d 天 X 值构成的时序数列的平均值除以标准差
    def ts_zcore4d(self, Matrix):
        ...

    # (X - delay(X, d))/delay(X, d)-1, 其中 delay(X, d) 为 X 在 d 天前的取值
    def ts_return4d(self, Matrix):
        ...

    # 过去 d 天 X 值构成的时序数列的加权平均值，权数为 d, d – 1, …, 1（权数之和应为 1, 需进行归一化处理）, 其中离现在越近的日子权数越大
    def ts_decaylinear4d(self, Matrix):
        ...

    # 过去 d 天 X 值构成的时序数列的平均值
    def ts_mean4d(self, Matrix):
        ...

~~~



AlphaNet 的代码实现

~~~python
class AlphaNet(nn.Module):
    def __init__(
        self, combination, combination_rev, index_list, fc1_num, fc2_num, dropout_rate
    ):
        super(AlphaNet, self).__init__()
        self.combination = combination
        self.combination_rev = combination_rev
        self.fc1_num = fc1_num
        self.fc2_num = fc2_num
        # 自定义的 Inception 模块
        self.Inception = Inception(combination, combination_rev, index_list)
        # 两个全连接层
        self.fc1 = nn.Linear(fc1_num, fc2_num)  # 702 -> 30
        self.fc2 = nn.Linear(fc2_num, 1)  # 30 -> 1
        # 激活函数
        self.relu = nn.ReLU()
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 xavier 的均匀分布对 weights 进行初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # 使用正态分布对 bias 进行初始化
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, data):
        data = self.Inception(data)  # N*702
        data = self.fc1(data)  # N*30
        data = self.relu(data)
        data = self.dropout(data)
        data = self.fc2(data)  # N*1
        # 线性激活函数，无需再进行激活
        data = data.to(torch.float)

        return data
~~~







问题：

感觉 AlphaNet的本质其实是一个“寻找一个 加入了 bn 的更稳定版本的 因子 参数 ” 的方法

大致方法其实可以是，有一个因子的序列











