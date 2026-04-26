import torch


def recall(scores, labels, k):
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    denominator = labels.sum(1).float().clamp(max=k).clamp_min(1)
    return (hit.sum(1).float() / denominator).mean().cpu().item()


def ndcg(scores, labels, k):
    effective_k = min(k, scores.size(1))
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :effective_k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + effective_k, device=scores.device)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    answer_count = labels.sum(1)
    ideal_count = answer_count.clamp(min=1, max=effective_k).long()
    idcg = torch.cumsum(weights, dim=0).gather(0, ideal_count - 1)
    idcg = torch.where(answer_count > 0, idcg, torch.ones_like(idcg))
    ndcg = dcg / idcg
    return ndcg.mean().cpu().item()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        effective_k = min(k, scores.size(1))
        cut = cut[:, :effective_k]
        hits = labels_float.gather(1, cut)
        denominator = answer_count.float().clamp(max=k).clamp_min(1)
        metrics['Recall@%d' % k] = (hits.sum(1) / denominator).mean().cpu().item()

        position = torch.arange(2, 2 + effective_k, device=scores.device)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        ideal_count = answer_count.clamp(min=1, max=effective_k).long()
        idcg = torch.cumsum(weights, dim=0).gather(0, ideal_count - 1)
        idcg = torch.where(answer_count > 0, idcg, torch.ones_like(idcg))
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics
