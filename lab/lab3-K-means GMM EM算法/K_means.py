import matplotlib.pyplot as plt
import numpy as np
import random


# 随机创建满足高斯分布的二维数据
# mu 方差, cov 协方差矩阵, data_num 数据个数
def create_gaussian_data(mu, cov, data_num):
    random_data = np.random.multivariate_normal(mu, cov, data_num).T
    return random_data


# 计算数据点到待定中心点的欧氏距离
# data 数据点, culster 待定中心点
def calcute_edistance(data, culster):
    result = None
    list = culster.tolist()
    for i in range(len(list)):
        t = data - np.mat(list[i])
        t = np.sum(np.square(t), axis=1)
        if result is None:
            result = t
        else:
            result = np.concatenate([result, t], axis=1)
    return result


# K mean算法
# data 数据点, culster 待定中心点
def k_mean(data, culster):
    # 各个聚类的中心点
    culster_new = culster

    while (True):
        dis = calcute_edistance(data, culster_new)
        # 各个聚类中的数据点
        culster_data = [[] for i in range(len(culster))]
        minindex = dis.argmin(axis=1).tolist()
        for i in range(len(minindex)):
            # 重新划分数据点
            culster_data[minindex[i][0]].append(data[i].tolist())
        culster_new_tmp = calcute_new_culster(culster_data)

        if (culster_new_tmp == culster_new).all():
            # 中心点不在发生变化
            return [culster_new, culster_data]
        else:
            culster_new = culster_new_tmp


# 计算新的中心点
def calcute_new_culster(culster_date):
    culster_new = []
    for i in range(len(culster_date)):
        aver = np.mean(np.mat(culster_date[i]), axis=0)
        culster_new.append(aver.tolist()[0])
    return np.mat(culster_new)


# 结果可视化
def show_result(k):
    data_A = create_gaussian_data(MU1, COV1, DATA_NUM)
    data_B = create_gaussian_data(MU2, COV2, DATA_NUM)
    data_C = create_gaussian_data(MU3, COV3, DATA_NUM)

    plt.subplot(1, 2, 1)
    plt.scatter(data_A[0], data_A[1], label="data_A = " + str(DATA_NUM) + ",aver = " + str(MU1) + ",cov = " + str(COV1))
    plt.scatter(data_B[0], data_B[1], label="data_B = " + str(DATA_NUM) + ",aver = " + str(MU2) + ",cov = " + str(COV2))
    plt.scatter(data_C[0], data_C[1], label="data_C = " + str(DATA_NUM) + ",aver = " + str(MU3) + ",cov = " + str(COV3))
    plt.title("orgin gussian distribute")
    plt.legend()

    plt.subplot(1, 2, 2)
    data = np.concatenate([data_A, data_B, data_C], axis=1).T

    x, culster = [], []
    for i in range(k):
        x.append(random.randint(0, len(data)))
        culster.append(data[x[i]])
    culster = np.mat(culster)
    result = k_mean(data, culster)
    center_point = result[0]
    data_point = result[1]
    for i in range(len(center_point)):
        tmp = np.mat(data_point[i]).T
        plt.scatter(tmp[0].tolist()[0], tmp[1].tolist()[0], label="culster = " + str(i))
    for i in range(len(center_point)):
        tmp = np.mat(center_point[i]).T
        plt.scatter(tmp[0].tolist()[0], tmp[1].tolist()[0], s=100, label="center = " + str(i) + ",position = ("
                    + "%.2f" %tmp[0].tolist()[0][0] + ", " + "%.2f" %tmp[1].tolist()[0][0] + ")")
    plt.title("k-mean cluster: K = " + str(k))
    plt.legend()
    plt.show()


# 二维高斯分布数据均值方差
MU1 = [20, 20]
MU2 = [10, 10]
MU3 = [10, 20]
COV1 = [[2, 0], [0, 2]]
COV2 = [[2, 0], [0, 2]]
COV3 = [[2, 0], [0, 2]]

# 二维高斯分布数据样本个数
DATA_NUM = 100

K = 6
show_result(K)
