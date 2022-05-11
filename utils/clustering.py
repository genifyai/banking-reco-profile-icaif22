from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# part of the code taken from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
def elbow_method(k_values, X):
    distortions = []
    inertias = []
    for k in k_values:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
    print("Distorsions")
    for k_val, val in zip(k_values, distortions):
        print(k_val, val)
    print("Inertias")
    for k_val, val in zip(k_values, inertias):
        print(k_val, val)
    fig, ax = plt.subplots()
    ax.plot(k_values, distortions, 'bx-')
    ax.set_xlabel('Values of K')
    ax.set_ylabel('Distortion')
    ax.set_title('Elbow Method using Distortion')
    fig, ax = plt.subplots()
    ax.plot(k_values, inertias, 'bx-')
    ax.set_xlabel('Values of K')
    ax.set_ylabel('Insertia')
    ax.set_title('Elbow Method using Inertia')
    plt.show()


# part of the code taken from https://www.geeksforgeeks.org/ml-principal-component-analysispca/
def print_pca(embedding, labels):
    pca = PCA(n_components=2)
    pca.fit(embedding)
    x_pca = pca.transform(embedding)
    print("Reduced to", x_pca.shape)
    plt.figure(figsize=(8, 6))

    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='plasma')

    # labeling x and y axes
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


def items_distribution_table(histories, items_name):
    n_clusters = len(histories)
    for i, item in enumerate(items_name):
        print(item+": ", end='')
        tot_items_row = 0
        for j in range(n_clusters):
            print(str(histories[j][item])+" ", end='')
            tot_items_row += histories[j][item]
        print("| ", end='')
        """
        computes the relative distribution of the items for each item category among clusters
        ex:
        #item1-cluster1 = 2
        #item1-cluster2 = 1
        #item1-cluster3 = 2
        result = [40% 20% 40%]
        item1 appears 5 times in total, 40% of the times in cluster1, 20% in cluster2 and 40% in cluster3
        """
        for j in range(n_clusters):
            if histories[j][item] == 0:
                print("0% ", end='')
            else:
                print("{:.3}% ".format(histories[j][item]/tot_items_row), end='')
        print("| ", end='')
        """
        computes the relative distribution of the items among item categories but in the same cluster
        ex:
        #item1-cluster1 = 0
        #item2-cluster1 = 1
        #item3-cluster1 = 2
        #item1-cluster2 = 1
        #item2-cluster2 = 1
        #item3-cluster2 = 3
        result = [0% 20%]  # distribution of item 1 in cluster 1 and 2
                 [33% 20%]  # distribution of item 2 in cluster 1 and 2
                 [66% 60%]  # distribution of item 3 in cluster 1 and 2
        """
        for j in range(n_clusters):
            tot_items_col = sum([histories[j][item_] for item_ in items_name])
            if histories[j][item] == 0:
                print("0% ", end='')
            else:
                print("{:.3}% ".format(histories[j][item] / tot_items_col), end='')
        print()


def items_distribution_chart(histories, items_name, relative=False):
    x_pos = [i for i in range(len(histories[0]))]
    if relative:
        tot_items = dict()
        for i, item in enumerate(items_name):
            tot_items[item] = sum([histories[c][item] for c in range(len(histories))])
    for c, history in enumerate(histories):
        fig, ax = plt.subplots(figsize=(15, 15))
        tot_items = sum([history[item] for item in items_name])
        if relative:
            items_history = [history[item]*100/tot_items for item in items_name]
        else:
            items_history = [history[item] for item in items_name]
        bar = ax.barh(x_pos, items_history, color='blue')
        if not relative:
            ax.set_xlabel('Item quantities in cluster {}'.format(c+1))
        else:
            ax.set_xlabel('(%) of items in cluster {}'.format(c+1))
        ax.set_yticks(x_pos)
        for i, v in enumerate(items_history):
            if v > 0:
                if relative:
                    ax.text(v + 1, i - 0.15, str("{:.4}".format(v)), color='black')
                else:
                    ax.text(v + 1, i - 0.15, str("{}".format(v)), color='black')
        #ax.bar_label(bar)
        #ax.set_ylabel('Items')
        #ax.set_title('Items distribution')

        ax.set_yticklabels(items_name)
        plt.show()