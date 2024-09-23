import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Critic model (renamed from Discriminator for WGAN)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img.view(-1, 784))

# Wasserstein loss
def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)

# Weight clipping function
def clip_weights(model, clip_value):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

# Training function
def train(generator, critic, device, train_loader, g_optimizer, c_optimizer, epoch):
    generator.train()
    critic.train()
    epoch_c_loss = 0
    epoch_g_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)

        # Train Critic
        for _ in range(5):  # Train critic more than generator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            real_loss = critic(real_imgs).mean()
            fake_loss = critic(fake_imgs.detach()).mean()
            c_loss = fake_loss - real_loss

            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

            # Clip critic weights
            clip_weights(critic, 0.01)

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        g_loss = -critic(fake_imgs).mean()

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_c_loss += c_loss.item()
        epoch_g_loss += g_loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]\tC_loss: {c_loss.item():.4f}\tG_loss: {g_loss.item():.4f}')

    c_losses.append(epoch_c_loss / len(train_loader))
    g_losses.append(epoch_g_loss / len(train_loader))


# Plot losses function
def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.plot(c_losses, label='Critic loss')
    plt.plot(g_losses, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.savefig('wgan_training_losses.png')
    plt.close()


# Image inpainting function
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


# Main function
def main():
    parser = argparse.ArgumentParser(description='WGAN MNIST Inpainting')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f'Use CUDA: {use_cuda}')
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
    critic = Critic().to(device)

    # Optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.00005)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=0.00005)

    # Training loop
    global c_losses, g_losses
    c_losses, g_losses = [], []
    for epoch in range(1, args.epochs + 1):
        train(generator, critic, device, train_loader, g_optimizer, c_optimizer, epoch)

    # Plot losses
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
