import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

color_map = {
    -1: (1., 1., 1.),
    0: (1.0, 0.0, 0.0),      # 红色
    1: (0.0, 1.0, 0.0),      # 绿色
    2: (0.0, 0.0, 1.0),      # 蓝色
    3: (0.0, 1.0, 1.0),      # 青色
    4: (1.0, 0.0, 1.0),      # 品红色
    5: (1.0, 1.0, 0.0),      # 黄色
    6: (0.5, 0.5, 0.5),      # 黑色
}


def draw_curve():
    tsne = TSNE(n_components=2, random_state=42)
    # target_tsne = tsne.fit_transform(target_representatives)
    feats_d = []
    feat_c = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    len_c = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    for i in [1,4,5]:
        feat = np.load('./nus_analysis/feat_' + str(i) + '.npy', allow_pickle=True).item()
        feats_d.append(feat)
        for j in range(7):
            feat_c[j].extend(feat[j][:500])
            len_c[j].append(len(feat[j][:500]))

    for i in range(7):
        colors = []
        for j in range(3):
            for k in range(len_c[i][j]):
                colors.append(color_map[j])uca
        # colors.append(color_map[j]*2000 for j in range(6))
        feat = np.array(feat_c[i])
        target_tsne = tsne.fit_transform(feat)
        plt.figure(figsize=(10, 7))
        plt.scatter(target_tsne[:, 0], target_tsne[:, 1], color=colors, label='Source Domain', alpha=0.6)
        plt.legend()
        plt.savefig('./nus_analysis/plot145_'+str(i)+'.png')
        print(i)


def distribution_plot():
    feats_d = []
    feat_c = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    feat_centroid = dict()
    for i in range(6):
        feat = np.load('./nus_analysis/feat_' + str(i) + '.npy', allow_pickle=True).item()
        feats_d.append(feat)
        feat_centroid[i] = [np.mean(feat[k], axis=0) for k in range(7)]
        for j in range(7):
            feat_c[j].extend(feat[j])

    distance = []
    for point in feats_d[0][0]:
            distance.append(np.linalg.norm(point-feat_centroid[0][0]))
    dist_mean_same = np.mean(np.array(distance), axis=0)
    dist_var_same = np.var(np.array(distance), axis=0)
    sns.kdeplot(distance, color='blue', label='same Domain', fill=True, alpha=0.5)

    dist_mean_diff = []
    dist_var_diff = []
    for i in range(1,6):
        distance = []
        # for point in feats_d[i][0]:
        #     distance.append(np.linalg.norm(point-feat_centroid[0][0]))
        distances = np.linalg.norm(target_pca - centroid, axis=1)
        # dist_mean_diff.append(np.mean(np.array(distance), axis=0))
        # dist_var_diff.append(np.var(np.array(distance), axis=0))
        sns.kdeplot(distances, color='red', label='diff Domain', fill=True, alpha=0.5)
        
    
    # x = np.arange(mean-20, mean+20, 0.1)
    # y1 = np.multiply(np.power(np.sqrt(2 * np.pi) * dist_var_same, -1), np.exp(-np.power(x - dist_mean_same, 2) / 2 * dist_var_same** 2))
    # # y = []
    # for i in range(5):
    #     y = np.multiply(np.power(np.sqrt(2 * np.pi) * dist_var_diff[i], -1), np.exp(-np.power(x - dist_mean_diff[i], 2) / 2 * dist_var_diff[i] ** 2))
    #     plt.plot(x, y)
    # plt.plot(x, y1, 'b-', linewidth=2)
    plt.show()
    # y2 = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma2, -1), np.exp(-np.power(x - u2, 2) / 2 * sigma2 ** 2))



if __name__ == '__main__':
    draw_curve()
    # distribution_plot()
