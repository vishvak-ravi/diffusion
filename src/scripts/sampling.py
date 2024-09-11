import torch
from numpy import sqrt
from configs.base_config import Config
from models import UNet

import torchvision.utils as vutils


def DDPM_sampling(net: UNet, config: Config, track_path=False, num_samples=128):
    with torch.no_grad():
        DDPM_config = config.ddpm_config
        initial_noise = torch.randn(
            (num_samples, config.img_channels, config.img_shape, config.img_shape)
        )  # initial
        betas = torch.linspace(
            DDPM_config.beta_min, DDPM_config.beta_max, DDPM_config.T
        )
        alphas = [1 - DDPM_config.beta_min]
        for t in range(1, DDPM_config.T):
            alpha_t = alphas[-1] * (1 - betas[t])
            alphas.append(alpha_t)

        denoise_path = []
        z_t = initial_noise
        for t in range(DDPM_config.T - 1, 2, -1):
            times = torch.full((num_samples,), t)  # expand t for batch
            alpha_t = alphas[t]
            mu = (
                1
                / (torch.sqrt(1 - betas[t]))
                * (z_t - betas[t] / torch.sqrt(1 - alphas[t]) * net(z_t, times))
            )
            if track_path and t % 10 == 0:
                denoise_path.append(mu[0])
                path_grid = vutils.make_grid(mu)
                vutils.save_image(path_grid, f"samples/path_{t}.jpg")
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


def heun_sampling(
    net: UNet,
    config: Config,
    track_path=False,
    num_samples: int = 10,
):
    with torch.no_grad():
        ## forward generation
        NUM_STEPS = 50
        sigma_max = 80
        sigma_min = 0.002
        rho = 7
        S_churn = 0  # apparently best for this method? idk if depends on VE/VP?
        S_min = 0
        S_noise = 1
        S_max = torch.inf
        class_labels = None
        step_indices = torch.arange(NUM_STEPS)

        timesteps = (
            sigma_max ** (1 / rho)
            + (step_indices / (NUM_STEPS - 1))
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        timesteps = torch.cat([timesteps, torch.zeros_like(timesteps[:1])])

        initial_noise = torch.randn(
            (num_samples, config.img_channels, config.img_size, config.img_size),
            requires_grad=False,
        )

        img_next = initial_noise * timesteps[0]  # loop initialization

        path = [img_next[0]]

        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            img_curr = img_next

            # increase noise temporarily
            gamma = (
                min(S_churn / NUM_STEPS, sqrt(2) - 1) if S_min <= t_curr <= S_max else 0
            )
            t_hat = torch.as_tensor(
                t_curr + gamma * t_curr
            )  # was a net.round_sigma here
            img_hat = img_curr + (
                t_hat**2 - t_curr**2
            ).sqrt() * S_noise * torch.randn_like(img_curr)

            img_hat = img_hat.float()
            t_hat = t_hat.float()

            # Euler step
            denoised = net(img_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (img_hat - denoised) / t_hat
            img_next = img_hat + (t_next - t_hat) * d_cur
            img_next = img_next.float()

            # 2nd order correction
            if i < NUM_STEPS - 1:
                denoised = net(img_next, t_next, class_labels).to(torch.float64)
                d_prime = (img_next - denoised) / t_next
                img_next = img_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        path.append(img_next[0])
    return img_next, torch.stack(path) if track_path else img_next
