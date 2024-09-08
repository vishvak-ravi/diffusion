import torch
import time
from torch.utils.tensorboard import SummaryWriter

from models import DiffusionNet
from configs.base_config import Config, DDPMConfig
from scripts.sampling import DDPM_sampling as sample


def train(model: DiffusionNet, config: Config, writer, save: bool = True):

    dataloader = config.get_dataloader()
    loss_fn = config.get_loss_fn()
    optimizer = config.get_optimizer(model.parameters())

    steps = 0

    for _ in range(config.epochs):
        for images, labels in dataloader:
            start = time.time()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, images, labels, return_samples=10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()ff
            steps += 1
            net_time = time.time() - start
            writer.add_scalar("time", net_time, steps)
            writer.add_scalar("loss", loss.item(), steps)
            if steps % config.log_interval == 0:
                print(f"logging things now!")
                sampled_img, an_img_path = sample(model, config, track_path=True)
                # writer.add_image("gt_img", gt_img, steps, dataformats="NCHW")
                # writer.add_image("noisy_gt", noisy_gt, steps, dataformats="NCHW")
                # writer.add_image("pred_img", pred_img, steps, dataformats="NCHW")
                writer.add_image("sampled_img", sampled_img, steps, dataformats="NCHW")
                writer.add_image("img_path", an_img_path, steps, dataformats="NCHW")
                print(f"logged {steps} steps for loss/imgs")
            if steps % config.grad_params_save_interval == 0:
                print(f"logging grad params now!")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"{name}_grad", param.grad, steps)
                    writer.add_histogram(f"{name}", param, steps)
                print(f"logged {steps} steps for grad/params")
                # sampled_img = wandb.Image(
                #     sampled_img, caption=f"Epoch: {epoch}, Step: {steps}"
                # )
                # an_img_path = wandb.Image(
                #     torch.stack(an_img_path), caption=f"Epoch: {epoch}, Step: {steps}"
                # )
                # wandb.log(
                #     {
                #         "loss": loss.item(),
                #         "epoch": epoch,
                #         "steps": steps,
                #         "img": sampled_img,
                #         "img_path": an_img_path,
                #     }
                # )
                # wandb.watch(model, log="all", log_freq=1)

    if save:
        torch.save(
            model.state_dict(),
            f"/mnt/meg/vishravi/diffusion/weights/{config.get_str()}.pth",
        )
        writer.close()


if __name__ == "__main__":
    # Print PYTHONPATH
    config = Config(
        "/mnt/meg/vishravi/diffusion/src/configs/cifar10_ddpm_single_class.json"
    )
    # wandb.init(
    #     project="diffusion",
    #     config={
    #         "config_id": {config.config_id},
    #     },
    # )
    writer = SummaryWriter(log_dir=f"src/scripts/logs/{time.asctime()}")
    writer.add_text("config", config.get_str(), 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(torch.cuda.get_device_name(device))
    model = DiffusionNet(config)
    # model.load_state_dict(
    #     torch.load("/mnt/meg/vishravi/diffusion/weights/default_128.pth")
    # )

    train(model, config, writer)
