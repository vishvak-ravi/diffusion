import torch
import wandb


class EDMLoss(torch.nn.Module):
    def __init__(self, config, log=True):
        super().__init__()
        self.P_mean = config.P_mean
        self.P_std = config.P_std
        if config.data == "cifar10":
            self.sigma_data = 0.5
        elif config.data == "mnist":  # TODO change values for mnist
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


class DDPMLoss(torch.nn.Module):
    def __init__(self, DDPM_config):
        super().__init__()
        self.beta_min = DDPM_config.beta_min
        self.beta_max = DDPM_config.beta_max
        self.T = DDPM_config.T
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.T)
        self.alphas = [1 - self.beta_min]
        for t in range(1, self.T):
            alpha = self.alphas[-1] * (1 - self.betas[t])
            self.alphas.append(alpha)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alphas = torch.Tensor(self.alphas).to(device=device)

    def forward(self, net, images, labels=None, return_samples: int = 0):
        sampled_timesteps = torch.randint(1, self.T, (images.shape[0],))
        sampled_noise = torch.randn(images.shape, device=images.device)
        sampled_alphas = torch.index_select(self.alphas, 0, sampled_timesteps)
        noisy_latent = (
            torch.sqrt(sampled_alphas.view(images.shape[0], 1, 1, 1)) * images
            + torch.sqrt(1 - sampled_alphas.view(images.shape[0], 1, 1, 1))
            * sampled_noise
        )
        pred_noise = net(noisy_latent, sampled_timesteps)
        loss = torch.norm(pred_noise - sampled_noise) ** 2
        gt_img = images
        noisy_gt = noisy_latent
        pred_img = noisy_latent - pred_noise
        return (
            (
                loss,
                gt_img[:return_samples],
                noisy_gt[:return_samples],
                pred_img[:return_samples],
            )
            if return_samples > 0 and return_samples < images.shape[0]
            else loss.sum()
        )
