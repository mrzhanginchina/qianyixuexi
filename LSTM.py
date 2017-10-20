import tensorflow as tf
import numpy as np


lstm_hidden_size = 10
batch_size = 10

# 通过一句话就可以声明一个lstm的机构，LSTM中使用的变量也会在该函数中自动声明。
lstm = tf.rnn_cell.BasicLSTMCell(lstm_hidden_size)
# 将LSTM中的状态初始化为全0数组，和其他神经网络一样，每次使用一个batch的训练样本。
state = lstm.zero_state(batch_size, tf.float32)
# 定义一个损失函数
loss = 0.0

# 在这里设置一个num_step来表示最长的可记忆的长度
#for i in range(num_steps):
    # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义的变量
    #if i>0:
     #   tf.get_variable_scope().resue_variables()
    # 每一步处理时间序列中的一个时刻，当前输入和前一时刻传入定义的LSTM结构可以得到当前的LSTM结构的
    # 输出lstm_output和更新后的状态state。
    #lstm_out, state = lstm(current_input, state)
    # 将当前时刻LSTM结构的输出得到一个全连接层，得到最终的输出。
    #final_out = fully_connected(lstm_output)
    # 计算当前时刻的输出的损失。
    #loss += calc_loss(final_out, expected_output)
