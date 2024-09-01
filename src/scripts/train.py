import torch, os

from models import DiffusionNet
from configs.base_config import Config


def train(model: DiffusionNet, config: Config, save: bool = True):

    dataloader = config.get_dataloader()
    loss_fn = config.get_loss_fn()
    optimizer = config.get_optimizer(model.parameters())

    for epoch in range(config.epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, images, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    if save:
        torch.save(model.state_dict(), f'../weights/{config["config_id"]}.pth')


if __name__ == "__main__":
    # Print PYTHONPATH
    config = Config("/mnt/meg/vishravi/diffusion/src/configs/prime.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    model = DiffusionNet(config).to(device)

    train(model, config)
