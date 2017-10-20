import numpy as np
X = [1, 2]
state = [0.0, 0.0]

# 分别定义不同的输入部分作为权重
# 这里是cell内部的状态权重
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
# 输入向量用到的权重
w_cell_input = np.asarray([0.5, 0.6])
# 这里的b_cell相当于是一个偏执。
b_cell = np.asarray([0.1, -0.1])

# 输出值的权重
w_output = np.asarray([[1.0], [2.0]])
# 用于输出的全连接层的参数
b_output = 0.1

for i in range(len(X)):
    # 在进行正向传播之前，首先需要计算state和w_cell_state的乘积(这一步是原来的原来的memory的残存)，然后把输入的第一个乘上w_cell_input，
    # 在加上b_cell，相当于是一个偏执。X[i]*w_cell_input之后是一个和w_cell_input相似的矩阵。
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    final_output = np.dot(state, w_output) + b_output

    print("before activation:", before_activation)
    print("state", state)
    print("output:", final_output)
