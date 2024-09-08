import torch
from configs.base_config import DDPMConfig


def DDPM_sampling(net, config, track_path=False, num_samples=128):
    with torch.no_grad():
        DDPM_config = config.ddpm_config
        initial_noise = torch.randn(
            (num_samples, config.img_channels, config.img_shape, config.img_shape)
        )  # initial
        betas = torch.linspace(
            DDPM_config.beta_min, DDPM_config.beta_max, DDPM_config.T
        )
        alphas = [1 - DDPM_config.beta_min]

        z_t = initial_noise
        for t in range(1, DDPM_config.T):
            alpha_t = alphas[-1] * (1 - betas[t])
            alphas.append(alpha_t)

        denoise_path = []

        for t in range(DDPM_config.T - 1, 2, -1):
            times = torch.full((num_samples,), t)  # expand t for batch
            alpha_t = alphas[t]
            mu = (
                1
                / (torch.sqrt(1 - betas[t]))
                * (z_t - betas[t] / torch.sqrt(1 - alphas[t]) * net(z_t, times))
            )
            if t % 10:
                denoise_path.append(mu[0])
            noise = torch.randn(
                (num_samples, config.img_channels, config.img_shape, config.img_shape)
            )
            z_t = mu + torch.sqrt(betas[t]) * noise  ## update z_t
        denoised = (
            1
            / torch.sqrt(1 - betas[0])
            * (
                z_t
                - betas[0]
                / torch.sqrt(1 - torch.tensor(alphas[0]))
                * net(z_t, torch.ones(num_samples))
            )
        )
        denoise_path.append(denoised[0])
        denoise_path = torch.stack(denoise_path)
        return denoised, denoise_path if track_path else denoised
