from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    TimeDistributed,
)
from metric import ccc
from loss import ccc_loss

# Model definition


class TestNet(pl.LightningModule):
    """
    Model to test whether everything works

    Input tensor shape: [batch_size, 3, 32, 32]
    Output tensor shape: [batch_size, 1]
    """

    def __init__(self, learning_rate=0.001):
        super().__init__()
        # Saving all arguments under hparams
        self.save_hyperparameters()
        # Network configuration
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Others configurations
        self.learning_rate = learning_rate  # Needed to auto detect best LR
        self.momentum = 0.9
        self.loss = torch.nn.CrossEntropyLoss()
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_train, y_train = batch
        # Applying the forward pass
        y_pred = self(x_train)
        # Loss assessing
        loss = self.loss(y_pred, y_train)
        # Metric assessing
        metric = self.metric(y_pred, y_train)
        # Logging values into the logger
        self.log("loss/train_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/train_metric", metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_dev, y_dev = batch
        # Applying the forward pass
        y_pred = self(x_dev)
        # Loss assessing
        loss = self.loss(y_pred, y_dev)
        # Metric assessing
        metric = self.metric(y_pred, y_dev)
        # Logging values into the logger
        self.log("loss/val_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/val_metric", metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_test, y_test = batch
        # Applying the forward pass
        y_pred = self(x_test)
        # Loss assessing
        loss = self.loss(y_pred, y_test)
        # Metric assessing
        metric = self.metric(y_pred, y_test)
        # Logging values into the logger
        self.log("loss/test_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/test_metric", metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum
        )
        return optimizer


class AlloSatNet(pl.LightningModule):
    """
    Model from '[Article] AlloSat - A New Call Center French Corpus for Satisfaction and Frustration Analysis'

    Input tensor shape: [batch_size, sequence_length, input_size]
    Output tensor shape: [batch_size, sequence_length, 1]
    """

    def __init__(self, input_size, batch_first=True, learning_rate=0.001):
        super().__init__()
        # Saving all arguments under hparams
        self.save_hyperparameters()
        # Network configuration
        self.bilstm1 = nn.LSTM(
            input_size, 200, bidirectional=True, batch_first=batch_first
        )
        self.bilstm2 = nn.LSTM(200 * 2, 64, bidirectional=True, batch_first=batch_first)
        self.bilstm3 = nn.LSTM(64 * 2, 32, bidirectional=True, batch_first=batch_first)
        self.bilstm4 = nn.LSTM(32 * 2, 32, bidirectional=True, batch_first=batch_first)
        self.dense1 = nn.Linear(32 * 2, 1)
        self.timedist = TimeDistributed(self.dense1, batch_first=batch_first)
        # Others configurations
        self.learning_rate = learning_rate  # Needed to auto detect best LR
        self.loss = ccc_loss
        self.metric = ccc

    def forward(self, x):
        x, _ = self.bilstm1(x)
        x = F.tanh(x)
        x, _ = self.bilstm2(x)
        x = F.tanh(x)
        x, _ = self.bilstm3(x)
        x = F.tanh(x)
        x, _ = self.bilstm4(x)
        x = F.tanh(x)
        x = self.timedist(x)
        return x

    def training_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_train, y_train = batch
        # Applying the forward pass
        y_pred = self(x_train)
        # Loss assessing
        loss = self.loss(y_pred, y_train)
        # Metric assessing
        metric = self.metric(y_pred, y_train)
        # Logging values into the logger
        self.log("loss/train_ccc_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/train_ccc", metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_dev, y_dev = batch
        # Applying the forward pass
        y_pred = self(x_dev)
        # Loss assessing
        loss = self.loss(y_pred, y_dev)
        # Metric assessing
        metric = self.metric(y_pred, y_dev)
        # Logging values into the logger
        self.log("loss/val_ccc_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/val_ccc", metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Getting the inputs and labels
        x_test, y_test = batch
        # Applying the forward pass
        y_pred = self(x_test)
        # Loss assessing
        loss = self.loss(y_pred, y_test)
        # Metric assessing
        metric = self.metric(y_pred, y_test)
        # Logging values into the logger
        self.log("loss/test_ccc_loss", loss, on_step=False, on_epoch=True)
        self.log("metric/test_ccc", metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
