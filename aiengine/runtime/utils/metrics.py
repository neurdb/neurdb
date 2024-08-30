from log import logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def roc_auc_compute_fn(y_pred, y_target):
    """IGNITE.CONTRIB.METRICS.ROC_AUC"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    if y_pred.requires_grad:
        y_pred = y_pred.detach()

    if y_target.is_cuda:
        y_target = y_target.cpu()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu()

    y_true = y_target.numpy()
    y_pred = y_pred.numpy()
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        logger.error(
            "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case"
        )
        return 0.0


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res
