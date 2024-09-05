import torch
import numpy as np

from models.ddpm import DiffusionNet
from configs.base_config import Config


def heun_sampling(
    net: DiffusionNet, config: Config, num_samples: int = 10, track_path=False
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
                min(S_churn / NUM_STEPS, np.sqrt(2) - 1)
                if S_min <= t_curr <= S_max
                else 0
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


def euler(net: DiffusionNet, config: Config, num_samples: int = 10, track_path=False):
    with torch.no_grad():
        ## forward generation
        NUM_STEPS = 100
        sigma_max = 80
        sigma_min = 0.002
        rho = 7
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

        initial_noise = torch.randn(
            (num_samples, config.img_channels, config.img_size, config.img_size),
            requires_grad=False,
        )

        img_next = initial_noise * timesteps[0]  # loop initialization

        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            img_curr = img_next

            # increase noise temporarily
            gamma = (
                min(S_churn / NUM_STEPS, np.sqrt(2) - 1)
                if S_min <= t_curr <= S_max
                else 0
            )
            t_hat = t_curr + gamma * t_curr  # was a net.round_sigma here
            img_hat = img_curr + (
                t_hat**2 - t_curr**2
            ).sqrt() * S_noise * torch.randn_like(img_curr)

            img_hat = img_hat.float()
            t_hat = t_hat.float()

            # Euler step
            denoised = net(img_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (img_hat - denoised) / t_hat
            img_next = img_curr + (t_next - t_hat) * d_cur
            img_next = img_next.float()

    return img_next


if __name__ == "__main__":
    config = Config("src/configs/attn.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    model = DiffusionNet(config)
    model.load_state_dict(
        torch.load("/mnt/meg/vishravi/diffusion/weights/default_128.pth")
    )
    model.to(device)
    # model.load_state_dict(torch.load(f'weights/{config["config_id"]}.pth'))
    import torchvision.utils as vutils

    # Call the heun_sampling function
    imgs, path = heun_sampling(model, config, num_samples=100, track_path=True)

    # Create a grid of images
    grid = vutils.make_grid(imgs, nrow=10, normalize=True)

    # Save the grid as an image
    vutils.save_image(grid, "gallery1.png")
