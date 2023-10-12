import cv2
import numpy as np
import matplotlib.pyplot as plt


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.001, max_iter=300):
        self.centers_ = {}
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 随机初始化聚类中心
        center_idx = np.random.randint(0, len(data), self.k_)
        for i in range(self.k_):
            self.centers_[i] = data[center_idx[i]]
        # 开始迭代
        for i in range(self.max_iter_):
            self.category = {}
            for i in range(self.k_):
                self.category[i] = []
            for dot in data:
                distances = []
                for center_idx in self.centers_:
                    # 使用欧氏距离
                    distances.append(np.linalg.norm(dot - self.centers_[center_idx]))
                classification = distances.index(min(distances))
                self.category[classification].append(dot)

            # 更新聚类中心
            prev_centers = dict(self.centers_)
            for c in self.category:
                self.centers_[c] = np.average(self.category[c], axis=0).astype(int)

            # 判断聚类结果是否满足要求
            optimized = True
            # 消除被除数为0的警告
            np.seterr(divide='ignore', invalid='ignore')
            for center_idx in self.centers_:
                org_centers = prev_centers[center_idx]
                cur_centers = self.centers_[center_idx]

                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    # 给定一个像素点 返回它应该属于的聚类中心
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return self.centers_[index]


if __name__ == '__main__':
    # 读取原始图像
    img = cv2.imread('scenery1.png')
    # print(img.shape)
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))

    nets = []
    for i in range(2,7):
        nets.append(K_Means(i))
    arrs = [data]
    for net in nets:
        img_copy = data.copy()
        net.fit(img_copy)
        for idx in range(len(img_copy)):
            img_copy[idx] = net.predict(img_copy[idx])
        arrs.append(img_copy)

    images = []
    for result in arrs:
        img = result.reshape(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=3',
              u'聚类图像 K=4', u'聚类图像 K=5', u'聚类图像 K=6']

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
