import argparse
from csv import reader
from utils.metrics import *


class User:
    def __init__(self, Id, recommendations=None, scores=None, ground_truth=None, ownership=None):
        self.Id = Id
        self.recommendations = recommendations
        self.scores = scores
        self.ground_truth = ground_truth
        self.ownership = ownership


if __name__ == '__main__':
    """
    predictions: json file containing the predictions given by amazon personalize
    ground_truth: csv file containing the ground truth recommendations of the last timestamp of each test user
    history: csv file containing the owned items during the N-1 timestamps
    ownership: bool, whether compute metrics on items ownership or acquisition
    it outputs the metrics relative to the recommendations predicted by Amazon Personalize
    USAGE: python amazon_personalize/results.py --predictions amazon_personalize/input_aws.json.out --ground_truth data/test_split.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--history', type=str, default="data/train_reduced.csv")
    parser.add_argument('--ownership', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    predictions = args.predictions
    ground_truth = args.ground_truth
    history = args.history
    ownership = args.ownership

    users = {}
    with open(history, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for row in csv_reader:
            if row[0] == "2016-04-28":
                user_id = int(float(row[1]))
                preds = row[26:]
                preds = [i for i, p in enumerate(preds) if len(p) > 0 and int(float(p)) == 1]
                users[user_id] = (User(user_id, ownership=preds))

    with open(predictions, 'r') as f:
        lines = f.readlines()
    for l in lines:
        user_id = int(l.split('{"input":{"userId":"')[1].split('"')[0])
        recs = l.split('{"recommendedItems":[')[1].split(']')[0].split('"')
        recs = [int(r) for r in recs if r not in [",", ""]]
        if not ownership:
            recs = [r for r in recs if r not in users[user_id].ownership]
        users[user_id].recommendations = recs

    n = 0
    with open(ground_truth, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for row in csv_reader:
            user_id = int(float(row[1]))
            preds = row[26:]
            if ownership:
                preds = [i for i, p in enumerate(preds) if int(float(p)) == 1]
            else:
                preds = [i for i, p in enumerate(preds) if int(float(p)) == 1
                         and i not in users[user_id].ownership]
            if len(preds) > 0:
                n += 1
            users[user_id].ground_truth = preds


    users_id = users.keys()
    n_users = n
    prec_1 = []
    prec_3 = []
    prec_5 = []
    prec_10 = []
    prec_20 = []
    rec_1 = []
    rec_3 = []
    rec_5 = []
    rec_10 = []
    mrr_20 = []
    ndcg_20 = []
    for i in users_id:
        preds = users[i].recommendations
        gt = users[i].ground_truth
        if len(gt) == 0:
            continue
        prec_1.append(precision_k(1, gt, preds))
        prec_3.append(precision_k(3, gt, preds))
        prec_5.append(precision_k(5, gt, preds))
        prec_10.append(precision_k(10, gt, preds))
        prec_20.append(precision_k(20, gt, preds))
        rec_1.append(recall_k(1, gt, preds))
        rec_3.append(recall_k(3, gt, preds))
        rec_5.append(recall_k(5, gt, preds))
        rec_10.append(recall_k(10, gt, preds))
        mrr_20.append(mrr_k(20, gt, preds))
        ndcg_20.append(ndcg_k(20, gt, preds))

    print("Precision 1:", np.sum(prec_1) / n_users)
    print("Precision 3:", np.sum(prec_3) / n_users)
    print("Precision 5:", np.sum(prec_5) / n_users)
    print("Precision 10:", np.sum(prec_10) / n_users)
    print("Precision 20:", np.sum(prec_20) / n_users)
    print("Recall 1:", np.sum(rec_1) / n_users)
    print("Recall 3:", np.sum(rec_3) / n_users)
    print("Recall 5:", np.sum(rec_5) / n_users)
    print("Recall 10:", np.sum(rec_10) / n_users)
    print("Mean Reciprocal Rank 20:", np.sum(mrr_20) / n_users)
    print("Normalized Discount Cumulative Gain 20:", np.sum(ndcg_20) / n_users)
