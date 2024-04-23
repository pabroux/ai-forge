import torch
import torchvision.transforms as transforms
import torchvision
import model as model
import loss as lossFunction
import metric as metric

""" 
Overall configuration
"""
path_model_to_save = "../../output/model/PytorchVanilla/cifar_net.pth"
path_tensorboard_to_save = "../../output/experimentManager/Tensorboard/cifar"
verbose = True
security = False


"""
Model
"""


def configure_model():
    return model.Net()


model = configure_model


"""
Data
"""
dataset = torchvision.datasets.CIFAR10

trainset_root = "../../../data"
trainloader_shuffle = True
trainloader_num_workers = 4

devset_root = "../../../data"
devloader_shuffle = False
devloader_num_workers = 4


"""
Preprocessing
"""
preprocessor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
preprocessor_target = None


"""
Loss
"""
loss = lossFunction.CrossEntropyLoss()


"""
Metrics
"""
metrics = {
    "CE": lossFunction.CrossEntropyLoss(),
}


"""
Optimizer (gradient descent)
"""


def configure_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


optimizer = configure_optimizer


"""
Early stopping
"""
earlystopping = True
earlystopping_min_delta = 0.001
earlystopping_patience = 2
earlystopping_restore_best_weights = True


"""
Training configuration
"""
epochs = 2
batch_size = 4
device = "cpu"
