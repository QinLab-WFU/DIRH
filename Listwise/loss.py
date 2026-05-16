import torch
import torch.nn.functional as F
from torch import nn


def calc_semantic_relevance(labels, metric="cosine", eps=1e-8):
    """
    Calculate semantic relevance matrix based on multi-hot labels.

    metric:
        cosine   : cosine similarity between normalized label vectors
        binary   : 0/1 relevance according to whether two samples share labels
        jaccard  : intersection-over-union of label sets
        euclidean: Euclidean-distance-based similarity
    """
    labels = labels.float()

    if metric == "cosine":
        normed_labels = F.normalize(labels, p=2, dim=1, eps=eps)
        relevance = normed_labels @ normed_labels.T

    elif metric == "binary":
        relevance = (labels @ labels.T > 0).float()

    elif metric == "jaccard":
        inter = labels @ labels.T
        label_count = labels.sum(dim=1, keepdim=True)
        union = label_count + label_count.T - inter
        relevance = inter / union.clamp_min(eps)

    elif metric == "euclidean":
        normed_labels = F.normalize(labels, p=2, dim=1, eps=eps)
        dist = torch.cdist(normed_labels, normed_labels, p=2)
        relevance = 1.0 - dist / (2.0 ** 0.5)
        relevance = relevance.clamp(0.0, 1.0)

    else:
        raise ValueError(f"Unknown semantic relevance metric: {metric}")

    return relevance.clamp(0.0, 1.0)


def calc_idcg(relevance):
    relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(2)
    relevance_indicator = (relevance_diff > 0).float()
    relevance_rank = torch.sum(relevance_indicator, dim=-1) + 1
    idcg = (2 ** relevance - 1) / torch.log2(1 + relevance_rank)
    return torch.sum(idcg, dim=-1)


def calc_dcg(scores, relevance, tau):
    scores_diff = scores.unsqueeze(1) - scores.unsqueeze(2)

    indicator = torch.sigmoid(scores_diff / tau)

    # self - self -> 0
    mask = torch.ones_like(scores)
    mask.fill_diagonal_(0)

    scores_rank = torch.sum(indicator * mask, dim=-1) + 1

    dcg = (2 ** relevance - 1) / torch.log2(1 + scores_rank)
    dcg = torch.sum(dcg, dim=-1)

    return dcg


class ListwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def fwd_t_loss(self, sim_mat, labels, margin=0.25):
        sames = labels @ labels.T > 0
        diffs = ~sames
        sames.fill_diagonal_(False)

        anc_idxes, pos_idxes, neg_idxes = torch.where(
            sames.unsqueeze(2) * diffs.unsqueeze(1)
        )

        S_ap = sim_mat[anc_idxes, pos_idxes]
        S_an = sim_mat[anc_idxes, neg_idxes]

        losses = F.relu(S_an - S_ap + margin)

        mask = losses > 0
        N = mask.sum()

        loss = 0 if N == 0 else losses[mask].mean()

        return loss, N

    def fwd_s_ndcg(self, sim_mat, labels, tau=0.01, relevance_metric="cosine"):
        scores = sim_mat

        # =====================================================
        # Original cosine relevance logic
        # 完全恢复原始 DIRH 的 cosine 逻辑
        # =====================================================
        if relevance_metric == "cosine":
            normed_labels = F.normalize(labels)
            relevance = normed_labels @ normed_labels.T  # belongs to [0, 1]

            idcg = calc_idcg(relevance)
            dcg = calc_dcg(scores, relevance, tau)

            ndcg = dcg / idcg

            loss = (1 - ndcg).mean()
            return loss

        # =====================================================
        # Other ablation variants
        # binary / jaccard / euclidean
        # =====================================================
        relevance = calc_semantic_relevance(
            labels,
            metric=relevance_metric
        )

        idcg = calc_idcg(relevance)
        dcg = calc_dcg(scores, relevance, tau)

        ndcg = dcg / idcg.clamp_min(1e-8)

        loss = (1 - ndcg).mean()
        return loss


if __name__ == "__main__":
    pass