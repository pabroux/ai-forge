import copy
import random
import torch
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl
from dataset import ESW1DataSet

# LightningDataModule definition


class TestDataModule(pl.LightningDataModule):
    """
    DataModule to test whether everything works
    """

    def __init__(self, batch_size):
        super().__init__()
        # Saving all arguments under hparams
        self.save_hyperparameters()
        self.batch_size = (
            batch_size  # Needed if we want to find the best batch_size automatically
        )
        # Preprocessor
        preprocessor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        preprocessor_target = None
        # Dataset
        dataset = torchvision.datasets.CIFAR10
        self.trainset = dataset(
            root="../../../data",
            train=True,
            transform=preprocessor,
            target_transform=preprocessor_target,
            download=True,
        )
        self.devset = dataset(
            root="../../../data",
            train=False,
            transform=preprocessor,
            target_transform=preprocessor_target,
            download=True,
        )
        # Dataloader options
        self.trainloader_shuffle = True
        self.trainloader_num_workers = 0
        self.devloader_shuffle = False
        self.devloader_num_workers = 0

    def train_dataloader(self):
        # Needed if we want to find the best batch_size automatically
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.trainloader_shuffle,
            num_workers=self.trainloader_num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.devset,
            batch_size=self.batch_size,
            shuffle=self.devloader_shuffle,
            num_workers=self.devloader_num_workers,
        )

    def test_dataloader(self):
        pass


class ESW1DataModule(pl.LightningDataModule):
    """
    DataModule for the ESW1 (Emotional Speech in the Wild 1) corpus
    """

    def __init__(
        self,
        batch_size=9,
        max_seq_len=7 * 60,
        dimension="Valence",
        gold_standard="Mean",
        normalize_input=False,
        normalize_target=False,
    ):
        super().__init__()
        # Asserting inputs
        assert dimension in [
            "Valence",
            "Dominance",
            "Arousal",
        ], "'dimension' should be either 'Valence', 'Dominance' or 'Arousal'."
        # Saving all arguments under hparams
        self.save_hyperparameters()
        self.batch_size = (
            batch_size  # Needed if we want to find the best batch_size automatically
        )
        # Dataset
        self.trainset = ESW1DataSet(
            path_input="../../../data/ESW1/Source/train",
            path_gold_standard=f"../../../data/ESW1/Gold Standard/Version/v0.0/{dimension}/{gold_standard}/Annotator1/train",
            max_seq_len=max_seq_len,
            padding=True,
            padding_circular=True,
            path_input_pattern="*.csv",
            path_gold_standard_pattern=dimension[0].upper() + "-*.csv",
            normalize_input=normalize_input,
            normalize_input_assess=True if normalize_input else False,
            normalize_target=normalize_target,
        )
        self.devset = ESW1DataSet(
            path_input="../../../data/ESW1/Source/dev",
            path_gold_standard=f"../../../data/ESW1/Gold Standard/Version/v0.0/{dimension}/{gold_standard}/Annotator1/dev",
            max_seq_len=max_seq_len,
            padding=True,
            padding_circular=True,
            path_input_pattern="*.csv",
            path_gold_standard_pattern=dimension[0].upper() + "-*.csv",
            normalize_input=normalize_input,
            normalize_target=normalize_target,
        )
        self.testset = ESW1DataSet(
            path_input="../../../data/ESW1/Source/test",
            path_gold_standard=f"../../../data/ESW1/Gold Standard/Version/v0.0/{dimension}/{gold_standard}/Annotator1/test",
            max_seq_len=max_seq_len,
            padding=True,
            padding_circular=True,
            path_input_pattern="*.csv",
            path_gold_standard_pattern=dimension[0].upper() + "-*.csv",
            normalize_input=normalize_input,
            normalize_target=normalize_target,
        )
        if normalize_input:
            self.devset.input_mean = self.trainset.input_mean
            self.devset.input_var = self.trainset.input_var
            self.testset.input_mean = self.trainset.input_mean
            self.testset.input_var = self.trainset.input_var
        # Dataloader options
        self.trainloader_shuffle = True
        self.trainloader_num_workers = 0
        self.devloader_shuffle = False
        self.devloader_num_workers = 0
        self.testloader_shuffle = False
        self.testloader_num_workers = 0

    def train_dataloader(self):
        # Needed if we want to find the best batch_size automatically
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.trainloader_shuffle,
            num_workers=self.trainloader_num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.devset,
            batch_size=self.batch_size,
            shuffle=self.devloader_shuffle,
            num_workers=self.devloader_num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=self.testloader_shuffle,
            num_workers=self.testloader_num_workers,
        )


class ESW1RandomDataModule(pl.LightningDataModule):
    """
    DataModule for the ESW1 (Emotional Speech in the Wild 1) corpus
    """

    def __init__(
        self,
        batch_size=9,
        max_seq_len=7 * 60,
        dimension="Valence",
        gold_standard="Mean",
        normalize_input=False,
        normalize_target=False,
        train_percentage=0.7,
    ):
        super().__init__()
        # Asserting inputs
        assert dimension in [
            "Valence",
            "Dominance",
            "Arousal",
        ], "'dimension' should be either 'Valence', 'Dominance' or 'Arousal'."
        # Saving all arguments under hparams
        self.save_hyperparameters()
        self.batch_size = (
            batch_size  # Needed if we want to find the best batch_size automatically
        )
        # Dataset
        self.trainset = ESW1DataSet(
            path_input="../../../data/ESW1/Source/all",
            path_gold_standard=f"../../../data/ESW1/Gold Standard/Version/v0.0/{dimension}/{gold_standard}/Annotator1/all",
            max_seq_len=max_seq_len,
            padding=True,
            padding_circular=True,
            path_input_pattern="*.csv",
            path_gold_standard_pattern=dimension[0].upper() + "-*.csv",
            normalize_input=normalize_input,
            normalize_input_assess=True if normalize_input else False,
            normalize_target=normalize_target,
        )
        self.devset = copy.deepcopy(self.trainset)
        self.testset = None
        base_list = (
            self.trainset.file_list_with_max_seq if max_seq_len else self.trainset.input
        )
        nb_file_to_select = int(len(base_list) * train_percentage)
        train_file_list = random.sample(base_list, nb_file_to_select)
        dev_file_list = [i for i in base_list if i not in train_file_list]
        self.trainset.file_list_with_max_seq = train_file_list
        self.devset.file_list_with_max_seq = dev_file_list

        if normalize_input:
            self.devset.input_mean = self.trainset.input_mean
            self.devset.input_var = self.trainset.input_var
        # Dataloader options
        self.trainloader_shuffle = True
        self.trainloader_num_workers = 0
        self.devloader_shuffle = False
        self.devloader_num_workers = 0

    def train_dataloader(self):
        # Needed if we want to find the best batch_size automatically
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.trainloader_shuffle,
            num_workers=self.trainloader_num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.devset,
            batch_size=self.batch_size,
            shuffle=self.devloader_shuffle,
            num_workers=self.devloader_num_workers,
        )

    def test_dataloader(self):
        pass
