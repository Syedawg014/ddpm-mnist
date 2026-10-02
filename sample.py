import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random
from scheduler import LinearNoiseScheduler
from modelArch import UNet
from dataset import get_dataloader


def sample(model, scheduler, device, num_samples=100, num_timesteps=1000):
    model.eval()

    xt = torch.randn((num_samples, 1, 28, 28), device=device)
    with torch.no_grad():
        for i in reversed(range(num_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            noise_pred = model(xt, t)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

            if (i + 1) % 100 == 0 or i == 0:
                images = torch.clamp(xt, -1.0, 1.0).cpu()
                images = (images + 1.0) / 2.0

                grid = make_grid(images, nrow=10).permute(1, 2, 0)

                plt.figure(figsize=(5, 5))
                plt.imshow(grid.numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"Timestep {i}")
                plt.show()


def testingImage(model,device):
    model.eval()
    train_loader,train_dataset = get_dataloader()
    sample_timestep = 350
    scheduler = LinearNoiseScheduler()
    randomimage = train_dataset[250][0].float().to(device).unsqueeze(0)
    plt.title("REAL IMAGE")
    plt.imshow(randomimage.squeeze(0).cpu().permute(1,2,0).numpy(),cmap="gray")
    plt.show()
    noise = torch.randn_like(randomimage).to(device)
    t = torch.full((1,), sample_timestep, device=device, dtype=torch.long)
    noisy_images = scheduler.add_noise(randomimage, noise, t)
    xt = noisy_images
    plt.title("IMAGE WITH PURE NOISE ADDED")
    plt.imshow(noisy_images.squeeze(0).cpu().permute(1,2,0).numpy(),cmap="gray")
    plt.show()
    with torch.no_grad():
        for i in reversed(range(sample_timestep)):
            t = torch.full((1,), i, device=device, dtype=torch.long)

            noise_pred = model(xt, t)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

            if (i + 1) % 100 == 0 or i == 0:
                images = torch.clamp(xt, -1.0, 1.0).cpu()
                images = (images + 1.0) / 2.0

                grid = make_grid(images, nrow=1).permute(1, 2, 0)

                plt.figure(figsize=(5, 5))
                plt.imshow(grid.numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"Timestep {i}")
                plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, t_emb_dim=128).to(device)
    scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    model.load_state_dict(torch.load('model.pth', map_location=device))
    sample(model, scheduler, device)
    testingImage(model,device)

if __name__ == '__main__':
    main()