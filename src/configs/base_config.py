import json
from types import SimpleNamespace

from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW

from models.losses import EDMLoss


class Config:
    def __init__(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
            self.__dict__.update(
                json.loads(
                    json.dumps(data), object_hook=lambda d: SimpleNamespace(**d)
                ).__dict__
            )

    def get_dataloader(self) -> DataLoader:
        if self.data == "cifar10":
            dataset = CIFAR10(
                root="data/",
                train=True,
                download=True,
                transform=Compose([ToTensor(), Normalize(0.5, 0.5)]),
            )
        elif self.data == "mnist":
            dataset = MNIST(
                root="data/",
                train=True,
                download=True,
                transform=Compose([ToTensor(), Normalize(0.5, 0.5)]),
            )
        else:
            raise ValueError(f"Unknown data: {self.data}")
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def get_loss_fn(self) -> EDMLoss:
        if self.loss == "edm":
            return EDMLoss(self)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def get_optimizer(self, params) -> Optimizer:
        if self.optimizer == "adamw":
            return AdamW(
                params,
                lr=self.lr,
                betas=tuple(self.betas),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


# Example usage:
# Config
# = Config
# ('Config
# .json')
# print(Config
# .some_attribute)
