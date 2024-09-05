import torch


def DDPM_sampling(net, config, num_samples=100):
    DDPM_config = config.ddpm_config
    initial_noise = torch.randn(
        (num_samples, config.img_channels, config.img_shape, config.img_shape)
    )  # initial
    betas = torch.arange(
        DDPM_config.beta_min,
        DDPM_config.beta_max,
        (DDPM_config.beta_max - DDPM_config.beta_min) // DDPM_config.T,
    )
    alphas = [1 - DDPM_config.beta_min]

    z_t = initial_noise
    for t in range(1, DDPM_config.T):
        alpha_t = alphas[-1] * (1 - DDPM_config.beta_min[t])
        alphas.append(alpha_t)

    for t in range(DDPM_config.T, 2, -1):
        times = torch.full((num_samples,), t)  # expand t for batch
        alpha_t = alphas[t]
        mu = (
            1
            / (torch.sqrt(1 - betas[t]))
            * (z_t - betas[t] / torch.sqrt(1 - alphas[t]) * net(z_t, times))
        )
        noise = torch.randn(
            (num_samples, config.img_channels, config.img_shape, config.img_shape)
        )
        z_t = mu + torch.sqrt(betas[t]) * noise  ## update z_t
    denoised = (
        1
        / torch.sqrt(1 - betas[0])
        * (
            z_t
            - betas[0] / torch.sqrt(1 - alphas[0]) * net(z_t, torch.ones(num_samples))
        )
    )
    return denoised
