import torch
import torch.nn.functional as F


'''
C, B, H, W, D:
    n_classes, batch_size, height, width, depth

logits(prediction):
    float in (-infty, infty)
    2D: [B, C, H, W], 3D: [B, C, H, W, D]

labels(answer):
    long integer
    2D: [B, H, W], 3D: [B, H, W, D]

'''


def mixed_dice_loss(logits, labels, *args):
    dice = dice_loss(logits, labels)
    ce = F.cross_entropy(logits, labels)
    return dice * ce + (1 - dice) * dice


def match_up(logits, labels, needs_softmax=True, batch_wise=False):

    probas = F.softmax(logits, dim=1) if needs_softmax else logits
    n_classes = logits.shape[1]

    # 2D: [B, C, H, W], 3D: [B, C, H, W, D]
    shape = logits.shape
    n_classes = shape[1]
    n_dim = len(shape[2:])

    # 2D: (0, 3, 1, 2), 3D: (0, 4, 1, 2, 3)
    permute_dim = (0, n_dim+1,) + tuple(i+1 for i in range(n_dim))

    # Batch-wise:     2D: (2, 3),    3D: (2, 3, 4)
    # Includes batch: 2D: (0, 2, 3), 3D: (0, 2, 3, 4)
    sum_dim = tuple(i+2 for i in range(n_dim))
    if not batch_wise:
        sum_dim = (0,) + sum_dim

    labels = F.one_hot(labels, n_classes).permute(permute_dim).float()
    assert probas.shape == labels.shape, (probas.shape, labels.shape)

    # # dice score without background
    # match = torch.sum(probas[:, 1:, ...] * labels[:, 1:, ...], sum_dim)
    # total = torch.sum(probas[:, 1:, ...] + labels[:, 1:, ...], sum_dim)

    match = torch.sum(probas * labels, sum_dim)
    total = torch.sum(probas + labels, sum_dim)

    return match, total


def compute_dice(match, total, eps=1e-10):
    return ((2. * match + eps) / (total + eps))


def dice_score(logits, labels, eps=1e-10, exclude_background=True):
    match, total = match_up(logits, labels, needs_softmax=True)
    multi_class_score = compute_dice(match, total, eps=eps)
    if exclude_background:
        return multi_class_score[1:]
    else:
        return multi_class_score


def dice_loss(logits, labels, weight=None):
    score = dice_score(logits, labels, exclude_background=True)
    if weight is not None:
        assert len(score) == len(weight), (len(score), len(weight))
        score *= weight
    return 1 - score.mean()


def balanced_dice_loss(logits, labels):
    n_labels = logits.shape[1]
    count = torch.zeros(n_labels-1).to(logits.device)
    for i in range(1, n_labels):
        count[i-1] = torch.sum(labels == i)
    weight = F.softmax(1/(count + 1e-10))
    return dice_loss(logits, labels, weight)


# def weighted_cross_entropy(weight):
#     assert type(weight) == list
#     _weight = torch.tensor(weight)
#     if torch.cuda.device_count() > 0:
#         _weight = _weight.cuda()

#     def loss(outputs, labels, *args):
#         return F.cross_entropy(outputs, labels, weight=_weight)

#     return loss


# def wce_dice(logits, labels, *args):
#     return wce(logits, labels) + balanced_dice_loss(logits, labels)

# def cross_entropy(logits, labels, fg_only=True):
#     if fg_only:
#         return F.cross_entropy(logits, labels, weight=WEIGHT)
#     else:
#         return F.cross_entropy(logits, labels)


def balanced_cross_entropy(logits, labels):
    n_labels = logits.shape[1]
    count = torch.zeros(n_labels).to(logits.device)
    total = torch.prod(torch.tensor(labels.shape))
    for i in range(1, n_labels):
        count[i] = torch.sum(labels == i)
        # assert count[i] > 0, i
    count[0] = total - torch.sum(count[1:])
    weight = F.softmax(1/(count + 1e-10))
    return F.cross_entropy(logits, labels, weight=weight)


class WeightedCrossEntropy:

    def __init__(self, n_labels=2):
        pass

    def __call__(self, logits, labels):
        n_labels = logits.shape[1]
        weight = torch.zeros(n_labels)
        total = torch.prod(torch.tensor(labels.shape))
        for i in range(1, n_labels):
            count = torch.sum(labels == i)
            # assert count > 0, i
            weight[i] = count
        # weight[0] = 1 - torch.sum(weight[1:])
        weight[0] = total - torch.sum(weight[1:])
        weight = F.softmax(weight) + 1e-7
        return F.cross_entropy(logits, labels, weight=weight)

# testing
# if __name__ == '__main__':
#     # 2D
#     a = torch.ones(2, 5, 5).long()
#     b = torch.randn(2, 3, 5, 5)
#     c = torch.zeros(2, 3, 5, 5)
#     c[:, 1, ...] = 100

#     # 3D
#     d = torch.ones(2, 5, 5, 6).long()
#     e = torch.randn(2, 3, 5, 5, 6)
#     f = torch.zeros(2, 3, 5, 5, 6)
#     f[:, 1, ...] = 100

#     print(
#         dice_score(b, a),
#         dice_score(c, a),
#     )
#     print(
#         dice_score(e, d),
#         dice_score(f, d),
#     )
