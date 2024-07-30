
import torch
import numpy as np

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class HugMetrics:

    class TopN:
        def __call__(self, eval_pred, mask_value=-100, topns=(1, 5, 10)):
            logits, labels = eval_pred
            bsize, seqlen = labels.shape

            logits = torch.from_numpy(np.reshape(logits, (bsize*seqlen, -1)))
            labels = torch.from_numpy(np.reshape(labels, (bsize*seqlen)))
            where_prediction = labels != mask_value

            topaccs = topk_accuracy(logits[where_prediction], labels[where_prediction], topk=topns)

            assert { n: acc for n, acc in zip(topns, topaccs) }