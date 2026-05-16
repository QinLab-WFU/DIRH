import torch
import torch.nn.functional as F

from Listwise.loss import calc_idcg as calc_idcg_mine, calc_dcg as calc_dcg_mine


def calc_idcg_code(relevance):
    relevance_repeat = relevance.unsqueeze(dim=2).repeat(1, 1, relevance.size(0))
    relevance_repeat_trans = relevance_repeat.permute(0, 2, 1)
    relevance_diff = relevance_repeat_trans - relevance_repeat
    relevance_indicator = torch.where(
        relevance_diff > 0, torch.full_like(relevance_diff, 1), torch.full_like(relevance_diff, 0)
    )
    relevance_rk = torch.sum(relevance_indicator, dim=-1) + 1
    idcg = (2**relevance - 1) / torch.log2(1 + relevance_rk)
    return torch.sum(idcg, dim=-1)


def sigmoid(tensor, tau=1.0):
    """temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / tau
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def calc_dcg_code(scores, relevance, tau=0.01):
    scores_repeat_t = scores.unsqueeze(dim=2).repeat(1, 1, scores.size(0))
    scores_repeat_trans_t = scores_repeat_t.permute(0, 2, 1)
    scores_diff_t = scores_repeat_trans_t - scores_repeat_t

    # image-to-text
    scores_sg_t = sigmoid(scores_diff_t, tau)

    mask = 1 - torch.eye(scores.shape[0], device=scores.device)
    scores_sg_t = scores_sg_t * mask
    scores_rk_t = torch.sum(scores_sg_t, dim=-1) + 1

    dcg_t = (2**relevance - 1) / torch.log2(1 + scores_rk_t)
    dcg_t = torch.sum(dcg_t, dim=-1)

    return dcg_t


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 4, 10, 8
    e, t, l = gen_test_data(B, C, K, is_multi_hot=True)

    _scores = e @ e.T
    _relevance = F.normalize(l) @ F.normalize(l).T

    print(calc_idcg_code(_relevance))
    print(calc_idcg_mine(_relevance))
    print("-" * 10)
    print(calc_dcg_code(_scores, _relevance))
    print(calc_dcg_mine(_scores, _relevance))
