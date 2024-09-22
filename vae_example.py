import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import argparse


# VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# training
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}: Average loss: {train_loss / len(train_loader.dataset):.4f}')


# make a partially missing image
def add_missing_part(image, device, missing_size=10):
    mask = torch.ones_like(image).to(device)
    mask[14:14+missing_size, 14:14+missing_size] = 0
    return image.to(device) * mask, mask


# try inpaint
def inpaint(model, device, image, mask):
    model.eval()
    with torch.no_grad():
        # 欠損画像をエンコード
        mu, logvar = model.encode(image.view(1, -1))
        # 潜在変数をサンプリング
        z = model.reparameterize(mu, logvar)
        # デコードして補完画像を生成
        reconstructed = model.decode(z)
        # マスクを使って元の画像と補完部分を結合
        inpainted = image.view(-1) * mask.view(-1) + reconstructed.view(-1) * (1 - mask.view(-1))
    return inpainted.view(28, 28)


# save model
def save_model(model, optimizer, epoch, filename):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, filename)
    print(f"Model saved to {filename}")


# load model
def load_model(model, optimizer, filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        print(f"Loaded model from {filename} (epoch {start_epoch})")
        return start_epoch
    return 0


# main
def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load saved model')
    parser.add_argument('--model-path', type=str, default='vae_model.pth',
                        help='path to save/load model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    batch_size = 128
    latent_dim = 20

    # データのロード
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # モデルの初期化
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    print(f'model path: {args.model_path}')
    if args.load_model:
        start_epoch = load_model(model, optimizer, args.model_path)

    # モデルのトレーニング
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train(model, device, train_loader, optimizer, epoch)

    # モデルの保存
    save_model(model, optimizer, end_epoch, args.model_path)

    # テスト画像の準備
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=1, shuffle=True)
    test_image = next(iter(test_loader))[0].squeeze()

    # 画像の一部を欠損させる
    missing_image, mask = add_missing_part(test_image, device)

    # 画像補完
    inpainted_image = inpaint(model, device, missing_image, mask)

    # 結果の表示
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
