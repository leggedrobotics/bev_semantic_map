import pytorch_lightning as pl
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import random
from torch.utils.data import Sampler
from itertools import chain


class MyBatchSampler(Sampler):
    def __init__(self, data_source):
        data_source = list(np.arange(1000))
        self.sequence_length = 20
        self.mini_seq_length = 2

        list_of_lists = [
            data_source[i : i + self.sequence_length]
            for i in range(0, len(data_source), self.sequence_length)
        ]

        self.data = list_of_lists  # list of sequences where each item of list contains sequence_length indices
        self.batch_size = 4
        self.num_sequences = 50
        self.num_samples = (
            self.num_sequences // self.batch_size
        ) * self.sequence_length
        self.num_samples = self.num_samples // self.mini_seq_length
        print(self.num_samples)

    def __iter__(self):
        random.shuffle(self.data)

        indices = list(chain(*self.data))
        indices = indices[: self.num_samples * self.batch_size * self.mini_seq_length]
        print(f"Number of Samples are {len(indices)}")
        print(f"Number of batches are {self.num_samples}")
        print(indices)

        for i in range(0, len(indices), self.batch_size * self.sequence_length):
            for j in range(i, i + self.sequence_length, self.mini_seq_length):
                yield [
                    indices[j + k * self.sequence_length]
                    for k in range(self.batch_size)
                ]

    def __len__(self):
        return self.num_samples


class SimpleDataset(Dataset):
    def __init__(self):
        X = np.arange(10000)
        y = X * 2
        X = [[_] for _ in X]
        y = [[_] for _ in y]
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "idx": idx}


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.criterion = MSELoss()

    def forward(self, inputs_id, labels=None):
        outputs = self.fc(inputs_id)
        loss = 0
        if labels is not None:
            loss = self.criterion(outputs, labels)
        return loss, outputs

    def train_dataloader(self):
        dataset = SimpleDataset()
        batch_sampler = MyBatchSampler(dataset)
        return DataLoader(dataset, num_workers=8, batch_sampler=batch_sampler)

    def training_step(self, batch, batch_idx):
        input_ids = batch["X"]
        labels = batch["y"]
        print(f"idx is {batch['idx']}")
        loss, outputs = self(input_ids, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == "__main__":
    model = MyModel()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)

    X = torch.Tensor([[1.0], [51.0], [89.0]])
    _, y = model(X)
    print(y)
