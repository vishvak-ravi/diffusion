import torch

from models.ddpm import DDPM
from configs.base_config import Config


def sample(net: DDPM, config: Config, num_samples: int = 10):
    ## forward generation
    NUM_STEPS = 100
    sigma_max = 80
    sigma_min = 0.002
    rho = 7
    LATENT_DIM = 100
    S_churn = 0  # apparently best for this method? idk if depends on VE/VP?
    S_min = 0
    S_noise = 1
    S_max = torch.inf
    class_labels = None
    step_indices = torch.arange(0, NUM_STEPS)

    timesteps = (
        sigma_max ** (1 / rho)
        + (step_indices / (NUM_STEPS - 1))
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    initial_noise = torch.randn(LATENT_DIM)

    img_next = initial_noise  # loop initialization

    for i, t_curr, t_next in enumerate(zip(timesteps[:-1], timesteps[1:])):
        img_curr = img_next

        # increase noise temporarily
        gamma = (
            min(S_churn / NUM_STEPS, torch.sqrt(2) - 1)
            if S_min <= t_curr <= S_max
            else 0
        )
        t_hat = net.round_sigma(t_curr + gamma * t_curr)
        img_hat = img_curr + (
            t_hat**2 - t_curr**2
        ).sqrt() * S_noise * torch.randn_like(img_curr)

        # Euler step
        denoised = net(img_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (img_hat - denoised) / t_hat
        img_next = img_curr + (t_next - t_hat) * d_cur

        # 2nd order correction
        if i < NUM_STEPS - 1:
            denoised = net(img_next, t_next, class_labels).to(torch.float64)
            d_prime = (img_next - denoised) / t_next
            img_next = img_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return img_next


if __name__ == "__main__":
    config = Config("src/configs/ddpm_config.json")
    model = DDPM(config)
    model.load_state_dict(torch.load(f'weights/{config["config_id"]}.pth'))
    sample(model)
