import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from pathlib import Path

def plot_dendogram(n_dim_data, target, dirpath):
    data_dist = pdist(n_dim_data[:, 0:len(n_dim_data[0]) - 1])  # computing the distance
    data_link = linkage(data_dist, method='ward', metric='euclidean')  # computing the linkage

    plt.figure(figsize=(20, 10))
    plt.xlabel(target)

    ddata = dendrogram(data_link, labels=n_dim_data[:, len(n_dim_data[0]) - 1], get_leaves=True)

    for i, d in zip(ddata['icoord'], ddata['dcoord']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        plt.plot(x, y, 'ro')
        plt.annotate("%.5g" % y, (x, y), xytext=(0, -8),
                     textcoords='offset points',
                     va='top', ha='center')
    path = dirpath / Path('hca_' + target + '.png')
    plt.savefig(path, dpi=600)
    print('file {0} saved.'.format(path))
