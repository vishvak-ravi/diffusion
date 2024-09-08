import json
from types import SimpleNamespace

from torchvision.datasets import CIFAR10, MNIST, Flowers102
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Subset
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW, Adam
from models.losses import EDMLoss, DDPMLoss


class Config:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, "r") as file:
            data = json.load(file)
            self.__dict__.update(
                json.loads(
                    json.dumps(data), object_hook=lambda d: SimpleNamespace(**d)
                ).__dict__
            )

            if self.data == "mnist":
                print("mnist")
                self.img_channels = 1
                self.img_size = 28
                self.data_centered = 0
            elif self.data == "cifar10":
                print("cifar10")
                self.img_channels = 3
                self.img_size = 32
                self.data_centered = 0

            if self.loss == "ddpm":
                self.ddpm_config = DDPMConfig(self.ddpm_config_path)

    def get_str(self) -> str:
        with open(self.json_path, "r") as file:
            return file.read()

    def get_dataloader(self) -> DataLoader:
        if self.data == "cifar10":
            dataset = CIFAR10(
                root="data/",
                train=True,
                download=False,
                transform=Compose([ToTensor(), Lambda(lambda x: x * 2 - 1)]),
            )
        elif self.data == "mnist":
            dataset = MNIST(
                root="data/",
                train=True,
                download=False,
                transform=Compose([ToTensor(), Normalize(0.5, 0.5)]),
            )
        else:
            raise ValueError(f"Unknown data: {self.data}")
        if self.classes != -1:
            dataset = Subset(
                dataset,
                [i for i in range(len(dataset)) if dataset.targets[i] in self.classes],
            )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        return dataloader

    def get_loss_fn(self) -> EDMLoss:
        if self.loss == "edm":
            return EDMLoss(self)
        elif self.loss == "ddpm":
            return DDPMLoss(self.ddpm_config)
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
        elif self.optimizer == "adam":
            return Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


class DDPMConfig:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, "r") as file:
            data = json.load(file)
            self.__dict__.update(
                json.loads(
                    json.dumps(data), object_hook=lambda d: SimpleNamespace(**d)
                ).__dict__
            )


# Example usage:
# Config
# = Config
# ('Config
# .json')
# print(Config
# .some_attribute)
