import torch
import wandb


class EDMLoss(torch.nn.Module):
    def __init__(self, config, log=True):
        super().__init__()
        if config.data == "cifar10":
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.5
        elif config.data == "mnist":  # TODO change values for mnist
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.3081
        else:
            raise ValueError(f"Unknown data: {config.data}")

    def forward(
        self, net, images, labels=None, augment_pipe=None, return_samples: int = 0
    ):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        loss = weight * ((D_yn - y) ** 2)

        return (
            (
                loss.sum(),
                y[:return_samples],
                (y + n)[:return_samples],
                D_yn[:return_samples],
            )
            if return_samples > 0 and return_samples < images.shape[0]
            else loss.sum()
        )
