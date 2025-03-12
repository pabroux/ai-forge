import torch
from torchmetrics.functional.regression import concordance_corrcoef

# Loss definition


def ccc_loss(y_pred, y_true):
    """
    Concordance correlation coefficient (CCC)-based loss function for a batch

    'y_pred' and 'y_true' shapes: [batch_size, sequence_length, 1]
    """
    y_pred = y_pred.squeeze(-1)
    y_true = y_true.squeeze(-1)
    ls = []
    for idx, _ in enumerate(y_pred):
        ls.append(concordance_corrcoef(y_pred[idx], y_true[idx]))
    ccc = torch.stack(ls).mean()
    return 1 - ccc
