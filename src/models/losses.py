import torch


class EDMLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.data == "cifar10":
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.5
        elif config.data == "mnist":  # TODO change values for mnist
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.5
        else:
            raise ValueError(f"Unknown data: {config.data}")

    def forward(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
