import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image


# 随机创建满足高斯分布的二维数据
# mu 方差, cov 协方差矩阵, data_num 数据个数
def create_gaussian_data(mu, cov, data_num):
    random_data = numpy.random.multivariate_normal(mu, cov, data_num).T
    return random_data


# 加载数据
# path 文件路径
# 返回数据X， 标签Y
def load_data():
    X = []
    Y = []
    # 读取文件
    file = open(DATA_PATH, 'r')
    try:
        text_lines = file.readlines()
        for line in text_lines:
            attr = line.strip().split(',')
            X.append([float(x) for x in attr[0: len(attr) - 1]])
            Y.append(attr[len(attr) - 1])

        # 解析数据内属性量，提取数据
        X = numpy.mat(X)
        Y = numpy.mat(Y).T
    finally:
        file.close()
    return [X, Y]


def pca(data, K):
    # 计算均值
    aver = numpy.mean(data, axis=0)
    # 计算协方差
    cov = numpy.cov((data - aver).T)
    # 计算该矩阵的特征值与特征向量
    x = numpy.linalg.eig(cov)
    # 将不同的特征值有小到大排列，将其索引记录表中
    args = numpy.argsort(-x[0])
    extra_aver = []
    effect = 0
    # 提取矩阵
    for i in range(K):
        extra_aver.append(aver.tolist()[args[i]])
        effect = effect + x[0][args[i]]
        if i == 0:
            extra_feature = x[1].T[args[i]:args[i]+1].T
        else:
            extra_feature = numpy.concatenate([extra_feature, x[1].T[args[i]:args[i]+1].T], axis=1)

    # 计算信噪比
    rate = abs(effect / (x[0].sum() - effect))

    print("信噪比=" + str(rate))
    result = numpy.dot((data - aver), extra_feature)
    # 重新映射到高维空间除燥
    recon = numpy.dot(result, extra_feature.T) + aver
    # 低维空间矩阵结果
    result = result + extra_aver
    return result, recon


# 结果可视化
def show_gussian_result():
    MU1 = [0, 0, 0]
    MU2 = [8, 8, 8]
    MU3 = [0, 0, 8]
    COV1 = [[2, 2, 0], [2, 2, 2], [0, 2, 1]]
    COV2 = [[2, 2, 0], [2, 2, 0], [0, 0, 1]]
    COV3 = [[2, 2, 0], [2, 2, 0], [0, 0, 1]]
    NUM = 100
    data_A = create_gaussian_data(MU1, COV1, NUM)
    data_B = create_gaussian_data(MU2, COV2, NUM)
    data_C = create_gaussian_data(MU3, COV3, NUM)

    data = numpy.concatenate([data_A, data_B, data_C], axis=1)
    ax = plt.subplot(121, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(data[0], data[1], data[2])
    plt.title("data picture")

    result = pca(data.T, K)[0].T
    plt.subplot(122)
    plt.scatter(result.tolist()[0], result.tolist()[1], label="culster = ")
    plt.title("PCA")
    plt.show()


def array_to_img(array):
    array=array*255
    new_img=Image.fromarray(array.astype(numpy.uint8))
    return new_img


def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col * each_width, row * each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_img(numpy.array(origin_imgs[i]).reshape(each_width, each_width))
        # 第二个参数为每次粘贴起始点的横纵坐标。在本例中，分别为（0，0）（28，0）（28*2，0）依次类推，第二行是（0，28）（28，28），（28*2，28）类推
        new_img.paste(each_img, ((i % col) * each_width, (int)(i / col) * each_width))
    return new_img


def mnist(k):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    imgs = mnist.train.images
    labels = mnist.train.labels

    origin_7_imgs = []
    for i in range(1000):
        if labels[i] == 7 and len(origin_7_imgs) < 100:
            origin_7_imgs.append(imgs[i])
    low_d_feat_for_7_imgs, recon_mat_for_7_imgs = pca(numpy.array(origin_7_imgs), k)  # 只取最重要的1个特征
    low_d_img = comb_imgs(recon_mat_for_7_imgs, 10, 10, 28, 28, 'L')
    low_d_img.show()


DATA_PATH = "data"
K = 2


show_gussian_result()
# mnist(1)


