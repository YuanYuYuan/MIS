import torch
import torch.nn.functional as F
import math
from medpy.metric import hd95

def wasserstein_distance(fake, real):
    return torch.mean(fake - real)

def wgan_generator_loss(fake):
    return -torch.mean(fake)


def mean_sqaure_loss(logits, labels):

    n_classes = logits.shape[1]
    probas = F.softmax(logits, dim=1)
    n_dim = len(logits.shape[2:])

    # 2D: (0, 3, 1, 2), 3D: (0, 4, 1, 2, 3)
    permute_dim = (0, n_dim+1,) + tuple(i+1 for i in range(n_dim))

    labels = F.one_hot(labels, n_classes).permute(permute_dim).float()
    assert probas.shape == labels.shape, (probas.shape, labels.shape)
    return F.mse_loss(probas, labels)


def housdorff_distance_95(logits, label):
    predis = F.softmax(logits, dim=1)
    n_classes = logits.shape[1]

    result = []
    for cls in range(1, n_classes):
        result.append(hd95(
            predis[:, cls, ...].squeeze(),
            label == cls
        ))

    return result


def domain_classification(logits):
    return torch.mean(torch.sigmoid(logits))

class BinaryDomainLoss:

    def __init__(self, label_smooth=True, label=1):
        assert label == 1 or label == 0
        self.label = label
        self.label_smooth = label_smooth
        self.target = None

    def __call__(self, x):
        x = torch.squeeze(x)
        if self.target is None or x.shape != self.target.shape:
            if self.label == 1:
                self.target = torch.ones(x.shape, device=x.device)
                if self.label_smooth:
                    self.target *= 0.9
            else:
                self.target = torch.zeros(x.shape, device=x.device)
                if self.label_smooth:
                    self.target += 0.1

        return F.binary_cross_entropy_with_logits(x, self.target)

class BinaryDomainAccu:

    def __init__(self, label=1):
        self.label = label
        self.target = None

    def __call__(self, x):
        x = torch.squeeze(x)
        if self.target is None or x.shape != self.target.shape:
            if self.label == 1:
                self.target = torch.ones(x.shape, device=x.device)
            else:
                self.target = torch.zeros(x.shape, device=x.device)

        return torch.mean(((torch.sigmoid(x) >= 0.5).float() == self.target).float())


class TwoDomainLoss:

    def __init__(self, label_smooth=True):
        self.truth = None
        self.label_smooth = label_smooth

    def __call__(self, x):
        x = torch.squeeze(x)
        assert x.requires_grad
        batch_size = x.shape[0]
        if self.truth is None or self.truth.shape != x.shape:
            self.truth = torch.zeros(x.shape, device=x.device)

            if self.label_smooth:
                self.truth[batch_size//2:] = 0.9
                self.truth[:batch_size//2] = 0.1
            else:
                self.truth[batch_size//2:] = 1.0

        return F.binary_cross_entropy_with_logits(x, self.truth)


class TwoDomainAccu:

    def __init__(self):
        self.truth = None

    def __call__(self, x):
        x = torch.squeeze(x)
        batch_size = x.shape[0]
        if self.truth is None or self.truth.shape != x.shape:
            self.truth = torch.zeros(x.shape, device=x.device)
            self.truth[batch_size//2:] = 1.0

        return torch.mean(((torch.sigmoid(x) >= 0.5).float() == self.truth).float())


class AdversarialLoss:

    def __init__(self, label_smooth=True):
        self.truth = None
        self.label_smooth = label_smooth

    def __call__(self, x):
        x = torch.squeeze(x)
        if self.truth is None or self.truth.shape != x.shape:
            self.truth = torch.ones(x.shape, device=x.device)
            if self.label_smooth:
                self.truth *= 0.9
        return F.binary_cross_entropy_with_logits(x, self.truth)


# this loss assumes the input is after sigomid
def adversarial_loss(x):
    return 1. - x.mean()


# this loss assumes the inputs are after sigomid
def discriminating_loss(x_true, x_fake):
    return x_true.mean() - x_fake.mean()


def VAE_KLD(latent_dist):
    mean, std = latent_dist
    std_square = std**2
    return torch.mean(mean**2 + std_square - torch.log(1e-10+std_square) - 1)


def L2_norm(x, y):
    return torch.mean((x - y)**2)

def VAE_L2(images, reconstructions):
    return torch.mean((images - reconstructions)**2)


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


def match_up(
    logits,
    labels,
    # mask=None,
    needs_softmax=True,
    batch_wise=False,
    threshold=0.
):

    # requires torch tensors
    assert isinstance(logits, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

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

    labels = F.one_hot(labels, n_classes).permute(permute_dim).contiguous().float()
    assert probas.shape == labels.shape, (probas.shape, labels.shape)

    # binarize the probas according to given threshold
    # NOTE: there might be multiple classes be 1
    if threshold > 0.:
        probas = (probas > threshold).float()

    # equivalent to apply argmax
    elif threshold == -1:
        probas = (probas > 1/n_classes).float()

    # if mask is None:
    #     match = torch.sum(probas * labels, sum_dim)
    #     total = torch.sum(probas + labels, sum_dim)
    # else:
    #     for idx, (s1, s2) in enumerate(zip(
    #         mask.shape,
    #         probas.shape
    #     )):
    #         if idx != 1:
    #             assert s1 == s2, (mask.shape, probas.shape)
    #     match = torch.sum(probas * labels * mask, sum_dim)
    #     total = torch.sum((probas + labels) * mask, sum_dim)

    match = torch.sum(probas * labels, sum_dim)
    total = torch.sum(probas + labels, sum_dim)
    return match, total


def compute_dice(match, total, smooth):
    # XXX
    # return ((2. * match + smooth) / (total + smooth))
    return ((2. * match) / (total + 1e-9))


def dice_score(
    logits,
    labels,
    smooth=1e-9,
    # mask=None,
    exclude_background=True,
    needs_softmax=True,
    threshold=0.,
    exclude_blank=False,
    batch_wise=False,
):
    if exclude_blank and labels.sum() == 0:
        multi_class_score = torch.tensor([math.nan]*logits.shape[1])

    else:
        match, total = match_up(
            logits,
            labels,
            needs_softmax=needs_softmax,
            threshold=threshold,
            batch_wise=batch_wise,
            # mask=mask,
        )
        multi_class_score = compute_dice(match, total, smooth=smooth)
        if batch_wise:
            multi_class_score = torch.mean(multi_class_score, dim=0)

    if exclude_background:
        return multi_class_score[1:]
    else:
        return multi_class_score


def dice_loss(
    logits,
    labels,
    weight=None,
    mask=None,
    exclude_background=True,
    needs_softmax=True,
    batch_wise=False,
    smooth=1e-9,
):
    score = dice_score(
        logits,
        labels,
        exclude_background=exclude_background,
        needs_softmax=needs_softmax,
        batch_wise=batch_wise,
        smooth=smooth,
    )

    if weight is not None:
        assert isinstance(weight, (list, tuple))
        assert len(score) == len(weight), (len(score), len(weight))
        weight = torch.tensor(weight).to(logits.device)
        score *= weight


    if mask is not None:
        assert isinstance(mask, (list, tuple))
        assert len(score) == len(mask), (len(score), len(mask))
        assert set(mask) == {0, 1}
        mask = torch.tensor(mask).to(logits.device)
        score *= mask
        return 1 - score.sum() / torch.count_nonzero(mask)
    else:
        return 1 - score.mean()

def pseudo_label(logits):
    return torch.argmax(logits, dim=1)


def confidence_mask(confidence_map, threshold=0.3, need_sigmoid=False):
    if need_sigmoid:
        return (torch.sigmoid(confidence_map) >= threshold).float()
    else:
        return (confidence_map >= threshold).float()


def spatial_masked_dice_loss(
    logits,
    labels,
    mask,
    exclude_background=True,
    batch_wise=False,
    smooth=1e-9,
):
    score = dice_score(
        logits,
        labels,
        mask=mask,
        exclude_background=exclude_background,
        batch_wise=batch_wise,
        smooth=smooth,
    )
    return 1 - score.mean()


def mask_predition(prediction, mask):
    return prediction * mask[None, :, None, None, None]

def masked_cross_entropy(logits, labels, mask=None):
    assert isinstance(mask, (list, tuple))
    assert set(mask) == {1, 0}
    assert logits.shape[1] == len(mask)
    mask = torch.Tensor(mask).to(logits.device)
    return F.cross_entropy(
        logits * mask[None, :, None, None, None],
        labels,
    )

def spatil_masked_cross_entropy(
    logits,
    labels,
    mask,
):
    loss = F.cross_entropy(
        logits,
        labels,
        reduction='none'
    )
    mask = torch.squeeze(mask)
    assert loss.shape == mask.shape, (loss.shape, mask.shape)
    return torch.mean(loss * mask)


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
