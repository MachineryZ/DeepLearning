# lstm
# 导入 pytorch 等工具包：
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# lstm 手写的结构：
# 对于lstm 内部的几个门控结构的公式

class NaiveCustomLSTM(nn.Module):
    def __init__(
        self,
        input_size: int, # 输入数据的维度
        hidden_size: int, # lstm 网络内部的结构
        output_size: int, # 网络输出的维度
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        """
        LSTM 公式
        先说明结构 
        forget gate 遗忘门, cell gate 细胞状态(网络隐藏层)
        output gate 输出门, 输入门 input gate

        变量含义：
        x_t: 第t时刻的输入, 维度是 [input_size]
        h_t: 第t时刻lstm内部的参数 hidden unit t, 维度是 [hidden_size]
        h_t-1: 第t-1时刻lstm内部的参数 hidden unit t-1, 维度同样是 [hidden_size]
        tanh(): 非线性激活函数, 双曲正切函数

        U_f, V_f, b_f: forget gate 的权重参数, 维度在下面定义会有
        f_t = sigmoid(U_f x_t + V_f h_t-1 + b_f) 遗忘门更新公式

        U_i, V_i, b_i: input gate 的权重参数, 维度在下面定义会有
        i_t = sigmoid(U_i x_t + V_i h_t-1 + b_i) 输入门更新公式
        
        U_o,V_o, b_o: output gate 的权重参数, 维度在下面定义会有
        o_t = sigmoid(U_o x_t + V_o h_t-1 + b_o) 输出们更新公式

        U_c, V_c, b_c: cell gate 的权重参数, 维度在下面定义会有
        g_t = sigmoid(U_c x_t + V_c h_t-1 + b_c) 细胞门更新公式
        c_t = f_t * c_t + i_t * g_t

        h_t = o_t * tanh(c_t) 更新t时刻的网络 hidden unit 参数
        
        """
        
        # 用代码描述出来上述公式：
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def init_weight(self):
        """
        参数初始化在深度神经网络的训练非常重要, 一组好的初始化参数会让网络更容易
        收敛到全局最优, 收敛速度更快。初始化参数尽量有随机性, 分布一致性即可
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        x: torch.Tensor, # 网络的输入值，可以使我们的因子, 也可以是 nlp 里面的词向量
        init_states = None, # 是否是第一个输入(0, 1, ..., T), 是否是第一个 
    ):
        batch_size, seq_len, _ = x.size() # 获取 x 的维度

        hidden_seq = [] # 存储序列

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_len): # 循环获取每 t 时刻的 x_t
            x_t = x[:, t, :] 
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = h_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            # 上面一大串公式，有一点要注意的是
            # @ 是矩阵乘法，比如 [3, 4] @ [4, 5] 是合法的矩阵乘法
            # * 是按位乘法，比如 [3, 4] * [3, 4] 是合法的，要求两个矩阵的维度是一样的

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        output = hidden_seq
        return hidden_seq.squeeze()

if __name__ == "__main__":
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    plt.figure()
    plt.plot(t[0:60], dataset[0:60,0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60,1], label = 'cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5') # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8') # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5 # Choose 80% of the data for testing
    train_data_len = int(data_len*train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # set batch size to 5
 
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
 
    lstm_model = NaiveCustomLSTM(
        input_size=INPUT_FEATURES_NUM, 
        hidden_size=16, 
        output_size=OUTPUT_FEATURES_NUM
    ) # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
 
    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files
 
    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval() # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
 
    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)

    plt.show()