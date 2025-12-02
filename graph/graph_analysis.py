from igraph import Graph
import leidenalg


def run_community(G):
    partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights='weight')

    return partition


