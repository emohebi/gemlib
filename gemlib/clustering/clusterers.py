from gemlib.abstarct.basefunctionality import BaseClusterer
from gemlib.visualization import hcaplot, networkplot
from sklearn.cluster import AffinityPropagation


class HierClustering(BaseClusterer):

    def run_algo(self):
        data = self.get_prepared_data()
        if not self.validate_data(data):
            return
        dirpath = self.dirpath + self.algo_name + '_filtered_' if self.filter is not None else self.dirpath + self.algo_name + '_'
        hcaplot.plot_dendogram(data, self.target, dirpath=dirpath)


class AffinityPropagationClustering(BaseClusterer):

    def run_algo(self):
        data = self.get_prepared_data()
        if not self.validate_data(data):
            return
        af = AffinityPropagation().fit(data[:, 0:len(data[0]) - 1])
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        if cluster_centers_indices is None:
            print('no clustering results ... terminating this clustering task.')
            return
        self.n_clusters = len(cluster_centers_indices)
        dirpath = self.dirpath + self.algo_name + '_filtered_' if self.filter is not None else self.dirpath + self.algo_name + '_'
        networkplot.plot_network(data[:, len(data[0]) - 1], labels, filename=self.target, dirpath=dirpath)
