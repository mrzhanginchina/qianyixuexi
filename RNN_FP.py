import numpy as np
X = [1, 2]
state = [0.0, 0.0]

# 分别定义不同的输入部分作为权重
# 这里是cell内部的状态权重
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
# 输入向量用到的权重
w_cell_input = np.asarray([0.5, 0.6])
# 还不清楚这里是什么作用
b_cell = np.asarray([0.1, -0.1])

# 输出值的权重
w_output = np.asarray([[1.0], [2.0]])
# 用于输出的全连接层的参数
b_output = 0.1

for i in range(len(X)):
    #
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    final_output = np.dot(state, w_output) + b_output

    print("before activation:", before_activation)
    print("state", state)
    print("output:", final_output)
