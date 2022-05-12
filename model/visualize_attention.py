import os
import torch
from genify_recosys.transformer import get_model
from utils.metrics import *
from torch.utils.data import Dataset
from genify_recosys.transformer_model import CustomDataset, logits_to_recs
from sklearn.cluster import KMeans
import numpy as np
from utils.clustering import *
from utils.metrics import *
import torch.nn.functional as F
import random
import seaborn as sns

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

# python genify_recosys/visualize_attention.py
if __name__ == '__main__':
    # those params should not be changed
    data_path = "data"
    n_items = 22
    d_model = 42
    heads = 7
    n_layers = 6
    length_history = 16
    weights_path = "genify_recosys/weights/"
    # those params can be changed
    limit_users = 10000  # int or None if we don't want to limit
    ownership = False  # compute results on products ownership (see blog for reference)
    random_users = [0, 5, 50, 99]
    random_sample_size = 100
    layers = [0, 2, 4, 5]
    show_labels = [0, 4]
    random.seed(10)
    # end params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(os.path.join(data_path, 'data.npz'))
    np.random.seed(1)
    x_test = data['x_test'][:, -length_history:]
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
    seq_weights = [[] for i in range(max(layers)+1)]
    print(seq_weights)
    prod_weights = [[] for i in range(max(layers)+1)]
    seq_score = [[] for i in range(max(layers)+1)]
    n_users = 0
    recommendations_tot, real_recommendations_tot = [], []
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits, att = model(batch, get_scores=True)
            for layer in layers:
                score = att[layer][1].detach()
                att_ = att[layer][0].detach()
                # print(type(att))
                # print(att.shape)
                # print("seq weights")
                x = torch.sum(att_, dim=-1)
                x = F.softmax(x, dim=-1)
                seq_weights[layer].append(x)
                x = torch.sum(score, dim=1)
                x = F.softmax(x, dim=-2)
                seq_score[layer].append(x)
                att_ = att_[:, :, -22:]
                x = torch.sum(att_, dim=1)
                x = F.softmax(x, dim=1)
                prod_weights[layer].append(x)

            j += 1
            if j == limit_users:
                break
    """seq_weights = [s.detach().numpy() for s in seq_weights]
    seq_weights = np.array(seq_weights).squeeze()
    seq_weights = np.mean(seq_weights, axis=0)
    print(seq_weights.shape)
    fig, ax = plt.subplots()
    bar = ax.bar(list(range(1, 17)), seq_weights, color='blue')
    ax.set_xlabel('# Month')
    ax.set_ylabel('Importance')
    plt.show()

    """
    for layer in layers:
        prod_weights_ = [s.detach().numpy() for s in prod_weights[layer]]
        prod_weights_ = np.array(prod_weights_).squeeze()
        """prod_weights = np.mean(prod_weights, axis=0)
        prod_weights = prod_weights[:][-22:]
        print(prod_weights.shape)
        fig, ax = plt.subplots(figsize=(15, 15))
        bar = ax.bar(list(range(1, 23)), prod_weights, color='blue')
        ax.set_xlabel('# Item')
        ax.set_ylabel('Importance')
        ax.set_xticks(list(range(1, 23)))
        ax.set_xticklabels(items, rotation=45)
        plt.show()

        for i in random_users:
            score = seq_score[i].squeeze()
            print(score.shape)
            fig, ax = plt.subplots(figsize=(15, 15))
            im = ax.imshow(score)
            ax.set_xticks(list(range(0, length_history)))
            ax.set_yticks(list(range(0, length_history)))
            ax.set_xticklabels(["Month {}".format(i) for i in range(1, length_history+1)])
            ax.set_yticklabels(["Month {}".format(i) for i in range(1, length_history+1)])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(length_history):
                for j in range(length_history):
                    text = ax.text(j, i, "{:.2}".format(score[i, j]),
                                   ha="center", va="center", color="w")
            fig.tight_layout()
            plt.show()
        """

        random_samples = random.sample(range(1, j), random_sample_size)
        fig, ax = plt.subplots(figsize=(5, 3))
        prod_weights_ = prod_weights_[random_samples].T
        ax = sns.heatmap(prod_weights_, cmap="YlGnBu", xticklabels=False)
        ax.set_yticks(list(range(0, 22)))
        #if layer in show_labels:
        #    ax.set_yticklabels(items, rotation=0)
        ax.set_xlabel('Users')
        fig.tight_layout()
        plt.show()