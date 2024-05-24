from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F


def compute_loss(scores, labels, margin=0.5):
    scores = 1-scores
    negs = scores[labels == 0]
    poss = scores[labels == 1]
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss


def compute_auc(scores, labels):
    scores = scores.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    return roc_auc_score(labels, scores)


def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    max_acc = 0
    best_threshold = -1

    positive_so_far = 0
    remaining_negatives = sum(labels == 0)

    for i in range(len(rows)-1):
        score, label = rows[i]
        if label == 1:
            positive_so_far += 1
        else:
            remaining_negatives -= 1

        acc = (positive_so_far + remaining_negatives) / len(labels)
        if acc > max_acc:
            max_acc = acc
            best_threshold = (rows[i][0] + rows[i+1][0]) / 2

    return max_acc, best_threshold
