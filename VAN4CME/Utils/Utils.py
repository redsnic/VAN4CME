import torch.nn.functional as F

def remap(t, range=(0, 100), scale=None):
    # return t  # no remap for now
    if scale is None:
        scale = range[1] - range[0]
    t = t / range[1]
    return (t * 2 - 1)*scale


def idx_to_onehot(idx, num_classes):
    return F.one_hot(idx.clamp(0, num_classes - 1).long(), num_classes).float()

