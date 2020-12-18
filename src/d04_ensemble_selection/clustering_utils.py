from rdkit.ML.Cluster import Butina
import numpy as np
import matplotlib.pyplot as plt


def get_cluster_centers(rmsmat: list, num: int, distThresh: float = 2.0) -> object:
    """
    Returns the indices of the cluster centers obtained with Butina clustering and the specific threshold.
    :param rmsmat: object from AllChem.GetConformerRMSMatrix
    :param num: number of conformers
    :param distThresh:
    :return:
    :rtype: list
    """
    rms_clusters = Butina.ClusterData(rmsmat, num, distThresh, isDistData=True, reordering=True)
    #print(f"Resulted in {len(rms_clusters)} clusters.")
    center_index = []
    for i in (rms_clusters):
        # print(f"Cluster of centroid {i[0]} has {len(i)} elements.")
        center_index.append(i[0])
    return center_index


def cluster_threshold_plot(rmsmat, num, min, max, stepsize):
    """Show plot to determine ideal threshold."""
    nclusters = []
    dist = []
    for n in np.arange(min, max, stepsize):
        cluster = Butina.ClusterData(rmsmat, num, n, isDistData=True, reordering=True)
        nclusters.append(len(cluster))
        dist.append(n)
    fig = plt.plot(dist, nclusters)
    plt.show()
    return fig