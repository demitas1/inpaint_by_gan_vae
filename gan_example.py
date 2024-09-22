import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse


# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(-1, 784))


# For plot training progress
d_losses = []
g_losses = []


# Training function
def train(generator, discriminator, device, train_loader, g_optimizer, d_optimizer, epoch):
    generator.train()
    discriminator.train()
    epoch_d_loss = 0
    epoch_g_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        real_loss = nn.BCELoss()(discriminator(real_imgs), torch.ones(batch_size, 1).to(device))
        fake_loss = nn.BCELoss()(discriminator(fake_imgs.detach()), torch.zeros(batch_size, 1).to(device))
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        g_loss = nn.BCELoss()(discriminator(fake_imgs), torch.ones(batch_size, 1).to(device))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]\tD_loss: {d_loss.item():.4f}\tG_loss: {g_loss.item():.4f}')

    # エポックごとの平均損失を記録
    d_losses.append(epoch_d_loss / len(train_loader))
    g_losses.append(epoch_g_loss / len(train_loader))


# Image inpainting
def inpaint(generator, device, image, mask):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated = generator(z).view(28, 28)
        inpainted = image * mask + generated * (1 - mask)
    return inpainted


# Add missing part to the image
def add_missing_part(image, device, missing_size=10):
    mask = torch.ones_like(image).to(device)
    mask[14:14+missing_size, 14:14+missing_size] = 0
    return image.to(device) * mask, mask


# 学習進行をプロットする
def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator loss')
    plt.plot(g_losses, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.savefig('gan_training_losses.png')
    plt.close()


# Main function
def main():
    parser = argparse.ArgumentParser(description='GAN MNIST Inpainting')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # Hyperparameters
    batch_size = 64
    global latent_dim
    latent_dim = 100

    # Data loading
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(generator, discriminator, device, train_loader, g_optimizer, d_optimizer, epoch)

    # 学習進行をプロット
    plot_losses()

    # Test image preparation
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=1, shuffle=True)
    test_image = next(iter(test_loader))[0].squeeze()

    # Add missing part to the image
    missing_image, mask = add_missing_part(test_image, device)

    # Inpainting
    inpainted_image = inpaint(generator, device, missing_image, mask)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image.cpu(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(missing_image.cpu(), cmap='gray')
    axes[1].set_title('Image with Missing Part')
    axes[2].imshow(inpainted_image.cpu(), cmap='gray')
    axes[2].set_title('Inpainted Image')
    plt.show()

if __name__ == '__main__':
    main()
