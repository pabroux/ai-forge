import torch.nn as nn

# Model weight initialization definition


def xavier_normal_initializer(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.xavier_normal_(param)
