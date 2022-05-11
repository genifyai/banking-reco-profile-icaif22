import os
import torch
from model.transformer import get_model
from utils.metrics import *
from torch.utils.data import Dataset
from model.transformer_model import CustomDataset, logits_to_recs
from sklearn.cluster import KMeans
import numpy as np
from utils.clustering import *

items = ['Current Accounts',
        'Derivada Account',
        'Payroll Account',
        'Junior Account',
        'Más particular Account',
        'particular Account',
        'particular Plus Account',
        'Short-term deposits',
        'Medium-term deposits',
        'Long-term deposits',
        'e-account',
        'Funds',
        'Mortgage',
        'Pensions 1',
        'Loans',
        'Taxes',
        'Credit Card',
        'Securities',
        'Home Account',
        'Payroll',
        'Pensions 2',
        'Direct Debit']

min_age = 2
max_age = 116
min_antiguedad = 3
max_antiguedad = 256
min_income = 7507.32
max_income = 11900871.51
segmento = ["Individuals", "College graduated", "VIP"]


# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts, key):
    if isinstance(list_dicts[0][key], (list, np.ndarray)):
        return np.concatenate([d[key] for d in list_dicts], axis=0)
    return [d[key] for d in list_dicts]


def inverse_scaler(v, vmin, vmax):
    return v * (vmax - vmin) + vmin


# python analyze_embedding.py
if __name__ == '__main__':
    # those params should not be changed
    data_path = "data"
    n_items = 22
    d_model = 42
    heads = 7
    n_layers = 6
    length_history = 16
    weights_path = "model/weights/genify_recosys.pth"
    # those params can be changed
    n_clusters = 5
    limit_users = None  # int or None if we don't want to limit
    use_elbow_method = False  # show elbow method results
    items_distributions_history = 1  # show history items distribution comparison among clusters
    items_distributions_preds = 1  # show predictions items distribution comparison among clusters
    pca = False
    ownership = False  # compute results on products ownership (see blog for reference)
    # end params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(os.path.join(data_path, 'data.npz'))
    np.random.seed(1)
    x_test = data['x_test']
    y_test = data['y_test']
    print("data loaded from", os.path.join(data_path, 'data.npz'))
    owned_items = None
    if not ownership:
        owned_items = []
        for i in range(x_test.shape[0]):
            owned_items.append(x_test[i][-1][-22:])
    test_set = CustomDataset(x_test, y_test)

    model = get_model(n_items, d_model, heads, 0,
                      n_layers, 2048, weights_path, device)
    print("model loaded from", weights_path)

    generator = torch.utils.data.DataLoader(
        test_set, batch_size=1
    )
    model.eval()
    j = 0
    users = []
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits, embedding = model(batch, get_embedding=True)
            embedding = embedding.detach().cpu().numpy().squeeze()
            recommendations = logits_to_recs(logits.detach().cpu().numpy())
            real_recommendations = [i for i, p in enumerate(labels[0].detach().cpu().numpy()) if int(float(p)) == 1]
            if owned_items is not None:
                # filter acquisition
                old_items = [i for i, p in enumerate(owned_items[j]) if int(float(p)) == 1]
                real_recommendations = [i for i in real_recommendations if i not in old_items]
                recommendations = [i for i in recommendations if i not in old_items]

            users.append(dict(embedding=embedding.flatten(),
                              items=real_recommendations,
                              preds=recommendations[:3],  # only top-3 preds
                              history=x_test[j][:, -22:],  # shape is (16, 22), 16 months, 22 items
                              age=x_test[j][0][18],
                              seniority=x_test[j][0][17],
                              income=x_test[j][0][19],
                              segmento=np.argmax(x_test[j][0][13:17]))
                         )
            j += 1
            if j == limit_users:
                break

    n_users = j
    embeddings = flatten_list_dicts(users, "embedding").reshape(-1, length_history * d_model)

    if use_elbow_method:
        print("Using elbow method to find optimum number of clusters...")
        elbow_method([2, 3, 4, 5, 6, 7, 8], embeddings)
        print("End elbow method")

    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)

    if pca:
        print_pca(embeddings, kmeans.labels_)

    cluster_histories = []
    cluster_recommendations = []
    for i in range(n_clusters):
        idx = np.where(np.array(kmeans.labels_) == i)
        cluster_users = [users[j] for j in idx[0]]
        print("-- Cluster", i + 1, "--------------------------")
        print("Users in cluster: {} - {:.3}%".format(len(cluster_users), len(cluster_users)*100/n_users))
        avg_age = inverse_scaler(np.mean(flatten_list_dicts(cluster_users, "age")), min_age, max_age)
        avg_sen = inverse_scaler(np.mean(flatten_list_dicts(cluster_users, "seniority")), min_antiguedad, max_antiguedad)
        avg_inc = inverse_scaler(np.mean(flatten_list_dicts(cluster_users, "income")), min_income, max_income)
        print("Avg age {}".format(avg_age))
        print("Avg seniority {}".format(avg_sen))
        print("Avg income {}".format(avg_inc))
        # analyze distributions
        segmento_dis = [0, 0, 0, 0]
        acquisitions = dict(zip(items, [0] * 22))
        history = dict(zip(items, [0] * 22))
        tot_acquisitions = 3 * len(cluster_users)
        tot_history = 0
        for user in cluster_users:
            segmento_dis[user["segmento"]] += 1
            for p in user["preds"]:
                acquisitions[items[p]] += 1
            for month in user["history"]:
                user_history = np.argwhere(month).flatten()
                if len(user_history) > 0:
                    for p in np.argwhere(month)[0]:
                        history[items[p]] += 1
                        tot_history += 1
        print("Segmento distribution")
        for s in range(3):
            print("- {}: {} - {:.3}%".format(str(segmento[s]), segmento_dis[s], segmento_dis[s]*100/len(cluster_users)))
        print("Acquisition distribution")
        cluster_recommendations.append(acquisitions)
        for j, (key, value) in enumerate(acquisitions.items()):
            print("{}) {}: {} - {:.3}%".format(j+1, key, value, value*100/tot_acquisitions))
        print("History distribution")
        cluster_histories.append(history)
        for j, (key, value) in enumerate(history.items()):
            print("{}) {}: {} - {:.3}%".format(j+1, key, value, value*100/tot_history))

    if items_distributions_history:
        print("Item distribution histories")
        items_distribution_table(cluster_histories, items)
        items_distribution_chart(cluster_histories, items)
        items_distribution_chart(cluster_histories, items, relative=True)
    if items_distributions_preds:
        print("Item distribution predictions")
        items_distribution_table(cluster_recommendations, items)
        items_distribution_chart(cluster_recommendations, items)
        items_distribution_chart(cluster_recommendations, items, relative=True)


"""
1) format the dataset in the format:
metadata + product history
2) computer embedding for each user
3) cluster embedding
4) compute stats for each cluster (try different numbers of clusters)
- avg age
- avg seniority
- avg income
- segmento distribution (how many students, ...)
- product history distribution
- acquired items distribution
"""