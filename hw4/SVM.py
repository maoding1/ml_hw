import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


### 展示结果
def plot_svc_decision_function(model, plot_support=True):
    """Plot the decision function for a 2D SVC"""

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # 绘制超平面
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig('./result.jpg')


if __name__ == '__main__':
    # 生成训练数据
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2],
              np.random.randn(20, 2) + [2, 2]]
    Y = [-1] * 20 + [1] * 20

    # 模型训练
    model = SVC(kernel='linear')
    model.fit(X, Y)

    # 结果可视化
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='jet')
    plot_svc_decision_function(model, plot_support=True)
