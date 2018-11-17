import matplotlib.pyplot as plt
import numpy as np
import math


# 随机创建满足高斯分布的二维数据
# mu 方差, cov 协方差矩阵, data_num 数据个数
def create_gaussian_data(mu, cov, data_num):
    random_data = np.random.multivariate_normal(mu, cov, data_num).T
    return random_data


# 加载数据
# path 文件路径
# 返回数据X， 标签Y
def load_data(path=None):
    X = []
    Y = []
    # 读取文件
    file = open(path[0], 'r')
    try:
        text_lines = file.readlines()
        for line in text_lines:
            attr = line.strip().split(',')
            X.append([float(x) for x in attr[0: len(attr) - 1]])
            Y.append(float(attr[len(attr) - 1]))

        # 解析数据内属性量，提取数据
        X = np.mat(X)
        Y = np.mat(Y).T
    finally:
        file.close()
    return [X, Y]


# 计算数据集的高斯分布
# sigma 协方差矩阵， u 均值， data 数据集
def guassian(sigma, u, data):
    tmp = data - u
    result = []
    sigma_det = pow(np.linalg.det(sigma) * pow(2 * math.pi, len(data.tolist()[0])), -0.5)
    for i in range(len(data)):
        result.append((tmp[i] * sigma.I * tmp[i].T).tolist()[0][0])
    result = sigma_det * np.exp(-0.5 * np.mat(result))
    return result


# Step E，计算后验概率γ
# pi 各个分布所占的比例， sigma协方差矩阵， u均值， data数据集
def posterior(pi, sigma, u, data):
    tmp = np.mat([0 for i in range(len(data))])
    for i in range(len(pi)):
        tmp = tmp + pi[i] * guassian(sigma[i], u[i], data)

    for i in range(len(pi)):
        temp = pi[i] * guassian(sigma[i], u[i], data) / tmp
        if i == 0:
            posterior = temp
        else:
            posterior = np.concatenate([posterior, temp], axis=0)

    return posterior.T

# 更新sigma
# u 均值， nk Nk，r_step Step E所得的后验概率， data 数据集
def update_sigma(u, nk, r_step, data):
    sigama = []
    r_step = r_step.T.tolist()
    nk = nk.tolist()[0]
    for i in range(len(u)):
        for j in range(len(data)):
            if j == 0:
                tmp = r_step[i][j] * (data[j] - u[i]).T * (data[j] - u[i])
            else:
                tmp = tmp + r_step[i][j] * (data[j] - u[i]).T * (data[j] - u[i])

        tmp = tmp * (1 / nk[i])
        sigama.append(tmp)
    return sigama



# EM算法
# data 数据集， iter_times 迭代次数， u 均值， sigma 方差， pi各个分布所占比例， label 是否有比对标签
def em_culster(data, iter_times, u, sigma, pi, label=None):
    i = 0
    while (i < iter_times):
        # Step E
        r_step = posterior(pi, sigma, u, data)
        # Step M
        nk = np.sum(r_step, axis=0)
        u = np.multiply((1 / nk).T, r_step.T * data)
        sigma = update_sigma(u, nk, r_step, data)
        pi = (nk / len(data)).tolist()[0]
        i = i + 1
        if label is not None:
            print(check_result(label, np.argmax(r_step, axis=1).tolist()))
    return [u, sigma, pi, r_step]


# 结果可视化
def show_result():
    data_A = create_gaussian_data(MU1, COV1, DATA_NUM1)
    data_B = create_gaussian_data(MU2, COV2, DATA_NUM2)
    data_C = create_gaussian_data(MU3, COV3, DATA_NUM3)
    plt.scatter(data_A[0], data_A[1], label="data_A = " + str(DATA_NUM1) + ",aver = " + str(MU1) + ",cov = " + str(COV1))
    plt.scatter(data_B[0], data_B[1], label="data_B = " + str(DATA_NUM2) + ",aver = " + str(MU2) + ",cov = " + str(COV2))
    plt.scatter(data_C[0], data_C[1], label="data_C = " + str(DATA_NUM3) + ",aver = " + str(MU3) + ",cov = " + str(COV3))
    plt.title("orgin gussian distribute")
    plt.legend()

    # 将元数据整合在一起
    data = np.concatenate([data_A, data_B, data_C], axis=1).T
    # 初始化pi， u， sigma
    pi = [0.7, 0.2, 0.1]
    u = np.mat([[0, 0], [0, 1], [1, 0]])
    sigma = [np.mat([[0.01, 0], [0, 0.01]]), np.mat([[0.01, 0], [0, 0.01]]), np.mat([[0.01, 0], [0, 0.01]])]

    # EM算法计算结果
    result = em_culster(data, 60, u, sigma, pi)
    # 提取计算所得的均值
    center_point = result[0].tolist()
    # 提取计算所得的协方差矩阵
    cov = result[1]
    # 提取计算所得的各个分布所占的比例
    pi = result[2]

    print(cov)
    print(pi)
    data_center = [[MU1[0], MU2[0], MU3[0]], [MU1[1], MU2[1], MU3[1]]]
    plt.scatter(data_center[0], data_center[1], label="data_center", marker="p", s=100)
    for i in range(len(center_point)):
        tmp = np.mat(center_point[i]).T
        plt.scatter(tmp[0].tolist()[0], tmp[1].tolist()[0], s=100,
                    label="center = " + str(i) + ",position = ("
                    + "%.2f" % tmp[0].tolist()[0][0] + ", "
                    + "%.2f" % tmp[1].tolist()[0][0] + ")")
    plt.legend()
    plt.show()


# 计算EM算法聚类的正确率
def check_result(label, result):
    true = 0
    for i in range(len(label)):
        if label[i][0] == result[i][0]:
            true = true + 1
    if true / len(label) > 0.5:
        return true / len(label)
    return 1 - true / len(label)


# 测试UCI的数据集来验证EM算法正确性
def test_uci_data():
    # 读取文件
    data = load_data(DATA_PATH)
    label = data[1].tolist()
    data = data[0]

    # 初始化各个参数
    cov = np.cov(data.T)
    pi = [0.7, 0.3]
    u = np.mat([[-2.2918, -7.257, 7.9597, 0.9211], [1.0987, 0.6394, 5.989, -0.58277]])
    sigma = [np.mat(cov), np.mat(cov)]
    # EM 算法聚类
    result = em_culster(data, ITER_TIMES, u, sigma, pi, label)
    # 测试效果
    check_result(label, np.argmax(result[3], axis=1).tolist())


# 二维高斯分布数据均值方差
MU1 = [0.6, 0.6]
MU2 = [0.1, 0.6]
MU3 = [0.6, 0.1]
COV1 = [[0.02, 0], [0, 0.02]]
COV2 = [[0.005, 0], [0, 0.005]]
COV3 = [[0.01, 0], [0, 0.01]]

# 二维高斯分布数据样本个数
DATA_NUM1 = 100
DATA_NUM2 = 100
DATA_NUM3 = 100
DATA_PATH = ['data']

ITER_TIMES = 50
show_result()
# test_uci_data()