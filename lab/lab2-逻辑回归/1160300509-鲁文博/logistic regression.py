import matplotlib.pyplot as plt
import numpy as np


# 随机创建满足高斯分布的二维数据
def create_gaussian_data(mu, cov, data_num):
    random_data = np.random.multivariate_normal(mu, cov, data_num).T
    return random_data


# 基于numpy的sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(x))


# 从分布中将x1, x2分离
def extra_data(tmp, label):
    result = []
    for t in range(len(tmp[0])):
        result.append([1, tmp[0][t], tmp[1][t], label])

    return result


# 加载数据
# path 文件路径
# 返回数据X， 标签Y， 初始系数向量W
def load_data(path=None):
    X, Y = [], []
    if path is None:
        # 如果没有文件路径，随机创建两个二维高斯分布
        training_positive_data = create_gaussian_data(MU1, COV1, POSITIVE_NUM)
        training_nagetive_data = create_gaussian_data(MU2, COV2, NEGATIVE_NUM)
        training_data = np.mat(extra_data(training_positive_data, 1) + extra_data(training_nagetive_data, 0))

        X = (training_data.T[0:3]).T
        Y = training_data.T[3].T
        W = np.mat([0 for i in range(3)])
    else:
        # 读取文件
        file = open(path[0], 'r')
        try:
            text_lines = file.readlines()
            for line in text_lines:
                attr = line.strip().split(',')
                X.append([1.0] + [float(x) for x in attr[0: len(attr) - 1]])
                Y.append(float(attr[len(attr) - 1]))

            # 解析数据内属性量，提取数据
            W = np.mat([0 for i in range(len(X[0]))])
            X = np.mat(X)
            Y = np.mat(Y).T
        finally:
            file.close()

    return [X, Y, W]


# 梯度下降法求解
# max_time 最大迭代次数， learning_rate 学习率，divisor 惩罚因子， path=None 文件路径，默认None
# 求解系数 W
def grad_des_optim(max_time, learning_rate, divisor, path=None):
    print("grad_des")
    par = load_data(path)
    X = par[0]
    Y = par[1]
    W = par[2]

    iter_time = 0
    while iter_time < max_time:
        Z = Y - (1 - sigmoid(X * W.T))
        update = np.sum(np.multiply(X, Z), axis=0)
        # 更新方程
        W = W + learning_rate * (update - divisor * W)

        if path is not None and iter_time % 100 == 0:
            # 有路径时输出loss查看分类情况
            loss = np.multiply(Y, X * W.T) - np.log(1 + np.exp(X * W.T))
            loss = np.sum(loss, axis=0)
            print("iter times = " + str(iter_time), end=" loss = ")
            print(loss)

        iter_time = iter_time + 1

    return W


# 牛顿法求解
# max_time 迭代次数, divisor 正则系数, path=None 文件路径，默认None
# 求解系数 W
def newton_optim(max_time, divisor, path=None):
    print("Newton")
    par = load_data(path)
    X = par[0]
    Y = par[1]
    W = par[2]

    iter_time = 0
    while iter_time < max_time:
        vec = np.multiply((1 - sigmoid(X * W.T)), sigmoid(X * W.T)).T.tolist()[0]
        A = np.diag(vec)
        # Hessian 矩阵
        H = X.T * A * X - divisor * np.eye(len(W.tolist()[0]))
        Z = X.T * (Y - (1 - sigmoid(X * W.T)))
        update = H.I * (Z - divisor * W.T)

        # 更新方程
        W = W + update.T

        if path is not None:
            # 有路径时输出loss查看分类情况
            loss = np.multiply(Y, X * W.T) - np.log(1 + np.exp(X * W.T))
            loss = np.sum(loss, axis=0)
            print("iter times = " + str(iter_time), end=" loss = ")
            print(loss)

        iter_time = iter_time + 1

    return W


# 计算 [Accuracy, Precision, Recall, F1-score]
# positive_data positive样本, negative_data negative 样本, weight 参数
def calculate_result(positive_data, negative_data, weight):
    par = [0, 0, 0, 0]     # TP TN FP FN
    positive_y = (positive_data[0] * (-weight[1]) - weight[0]) / weight[2]
    negative_y = (negative_data[0] * (-weight[1]) - weight[0]) / weight[2]

    for i in range(len(positive_data[0])):
        if positive_y[i] < positive_data[1][i]:
            par[0] = par[0] + 1
        else:
            par[2] = par[2] + 1

    for i in range(len(negative_data[0])):
        if negative_y[i] > negative_data[1][i]:
            par[1] = par[1] + 1
        else:
            par[3] = par[3] + 1

    return [(par[0] + par[1]) / (par[0] + par[1] + par[2] + par[3]) * 100,
            par[0] / (par[0] + par[2]) * 100,
            par[0] / (par[0] + par[3]) * 100,
            par[0] * 2 / (2 * par[0] + par[2] + par[3]) * 100]


# 计算UCI数据库数据分类效果 [Accuracy, Precision, Recall, F1-score]
# weight 计算出的预测系数, path 测试文件路径
def calculate_effort(weight, path):
    X, Y = [], []
    # 读取文件
    file = open(path[1], 'r')
    try:
        text_lines = file.readlines()
        for line in text_lines:
            attr = line.strip().split(',')
            X.append([1.0] + [float(x) for x in attr[0: len(attr) - 1]])
            Y.append(float(attr[len(attr) - 1]))

        # 解析数据内属性量，提取数据
        X = np.array(X)
        pre = (X * weight.T).T.tolist()[0]
    finally:
        file.close()

    par = [0, 0, 0, 0]  # TP TN FP FN
    for i in range(len(Y)):
        if Y[i] == 1 and pre[i] > 0:
            par[0] = par[0] + 1
        if Y[i] == 1 and pre[i] < 0:
            par[3] = par[3] + 1
        if Y[i] == 0 and pre[i] > 0:
            par[2] = par[2] + 1
        if Y[i] == 0 and pre[i] < 0:
            par[1] = par[1] + 1

    effect = [(par[0] + par[1]) / (par[0] + par[1] + par[2] + par[3]) * 100,
              par[0] / (par[0] + par[2]) * 100,
              par[0] / (par[0] + par[3]) * 100,
              par[0] * 2 / (2 * par[0] + par[2] + par[3]) * 100]
    # 输出四项指标
    print("Acc = " + '%.1f' % effect[0] + "%, Prec = " + '%.1f' % effect[1] + "%, Recall = " \
          + '%.1f' % effect[2] + "%, F1_s = " + '%.1f' % effect[3] + "%")
    return


# 结果可视化
# optim_type 优化类型——0：梯度下降，1：牛顿法，path=None 文件路径，默认None
def show_result(optim_type, path=None):
    if optim_type == 0:
        weight = grad_des_optim(ITER_TIMES, LR, DIVISOR, path)
    else:
        weight = newton_optim(NEWTON_ITER, DIVISOR, path)

    if path is not None:
        # 当有文件路径时，计算测试效果
        calculate_effort(weight, path)
        return
    # 测试集
    weight = weight.tolist()[0]
    test_positive_data = create_gaussian_data(MU1, COV1, POSITIVE_NUM)
    test_nagetive_data = create_gaussian_data(MU2, COV2, NEGATIVE_NUM)
    plt.scatter(test_positive_data[0], test_positive_data[1], label="positive = " + str(POSITIVE_NUM))
    plt.scatter(test_nagetive_data[0], test_nagetive_data[1], label="negative = " + str(NEGATIVE_NUM))

    effect = calculate_result(test_positive_data, test_nagetive_data, weight)
    data1 = np.linspace(0, 30, 100)
    data2 = (-weight[0] - weight[1] * data1) / weight[2]
    plt.plot(data1, data2, label="y = " + '%.2f' % (-weight[1] / weight[2]) + "x + "
                                 + '%.2f' % (-weight[0] / weight[2]))
    if optim_type == 0:
        title = "LR = " + str(LR) + ",  ITER_TIMES = " + str(ITER_TIMES) + ",  λ = " + str(DIVISOR) \
                + "\nAcc = " + '%.1f' % effect[0] + "%, Prec = " + '%.1f' % effect[1] + "%, Recall = " \
                + '%.1f' % effect[2] + "%, F1_s = " + '%.1f' % effect[3] + "%"
    else:
        title = "ITER_TIMES = " + str(NEWTON_ITER) + ",  λ = " + str(DIVISOR) \
                + "\nAcc = " + '%.1f' % effect[0] + "%, Prec = " + '%.1f' % effect[1] + "%, Recall = " \
                + '%.1f' % effect[2] + "%, F1_s = " + '%.1f' % effect[3] + "%"

    plt.title(title)
    plt.legend()
    plt.show()


# 二维高斯分布数据均值方差
# MU1 = [17, 17]
# MU2 = [12, 12]
MU1 = [20, 20]
MU2 = [10, 10]
COV1 = [[10, 0], [0, 10]]
COV2 = [[10, 0], [0, 10]]

# 二维高斯分布数据样本个数
POSITIVE_NUM = 100
NEGATIVE_NUM = 100

# 学习率、惩罚因子、梯度下降法与牛顿法迭代次数
LR = 0.001
DIVISOR = 0
ITER_TIMES = 50000
NEWTON_ITER = 15

NEWTON_OPTIM = 1
GRAD_DES_OPTIM = 0
DATA_PATH = ['training data.txt', 'test data.txt']

show_result(GRAD_DES_OPTIM)
