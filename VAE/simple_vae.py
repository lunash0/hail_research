import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2


# dataset class
class dataset(Dataset):
    def __init__(self, data):
        self.x = data["numpy"]
        self.y = data["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        single_x = self.x[index]
        single_y = self.y.iloc[index]
        single_y = torch.tensor(single_y)
        self.augmentor = self.augmentation()
        augmented = self.augmentor(image=single_x)["image"]

        normalizated = self.normalization(augmented)
        return normalizated, single_y

    def augmentation(self):
        pipeline = []
        pipeline.append(A.HorizontalFlip(p=0.5))
        pipeline.append(A.RandomBrightnessContrast(p=0.5))
        pipeline.append(A.Resize(height=200, width=200))
        pipeline.append(ToTensorV2(transpose_mask=True, p=1))

        return A.Compose(pipeline, p=1)

    def normalization(self, array):
        normalizated_array = (array - array.min()) / (array.max() - array.min())

        return normalizated_array


def split_feature_label(file_path):
    label = file_path.split("\\")[1]
    img = Image.open(file_path).convert("L")
    img_array = np.array(img, dtype=np.uint8)
    return img_array, label


def fileopen_splitdata_labeling(dir):
    train_data = {"numpy": [], "label": []}
    val_data = {"numpy": [], "label": []}
    test_data = {"numpy": [], "label": []}
    train_shape = np.zeros(2)
    val_shape = np.zeros(2)
    test_shape = np.zeros(2)

    test_path = []
    for (directory, _, filenames) in os.walk(dir):
        if "Training" in directory:
            for filename in filenames:
                if ".jpg" in filename:
                    file_path = os.path.join(directory, filename)
                    img_array, label = split_feature_label(file_path=file_path)
                    train_shape = train_shape + img_array.shape
                    train_data["numpy"].append(img_array)
                    train_data["label"].append(label)

        elif "Validation" in directory:
            for filename in filenames:
                if ".jpg" in filename:
                    file_path = os.path.join(directory, filename)
                    img_array, label = split_feature_label(file_path=file_path)
                    val_shape = val_shape + img_array.shape
                    val_data["numpy"].append(img_array)
                    val_data["label"].append(label)

        elif "Testing" in directory:
            for filename in filenames:
                if ".jpg" in filename:
                    file_path = os.path.join(directory, filename)
                    test_path.append(file_path)
                    img_array, label = split_feature_label(file_path=file_path)
                    test_shape = test_shape + img_array.shape
                    test_data["numpy"].append(img_array)
                    test_data["label"].append(label)

    train_data["label"] = pd.get_dummies(train_data["label"])
    val_data["label"] = pd.get_dummies(val_data["label"])
    test_data["label"] = pd.get_dummies(test_data["label"])

    train_avg_shape = train_shape / len(train_data["numpy"])
    val_avg_shape = val_shape / len(val_data["numpy"])
    test_avg_shape = test_shape / len(test_data["numpy"])

    print(f"\n\n###ë°ì´í° ì ë³ íê·  ì´ë¯¸ì§ í¬ê¸° íì¸###\n!!!!WARNING!!!!")
    print(f"1. trainë°ì´í° íê·  ì´ë¯¸ì§ í¬ê¸° : {train_avg_shape}")
    print(f"2. valë°ì´í° íê·  ì´ë¯¸ì§ í¬ê¸° : {val_avg_shape}")
    print(f"3. testë°ì´í° íê·  ì´ë¯¸ì§ í¬ê¸° : {test_avg_shape}")

    return train_data, val_data, test_data, test_path


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(18432, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cpu()
        self.N.scale = self.N.scale.cpu()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        logvar = self.linear3(x)
        sigma = torch.exp(logvar) ** 0.5

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return z


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 18432),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 24, 24))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels=16, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)


### Training function
def train_epoch(vae, device, dataloader, optimizer):
    vae.train()
    train_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)
        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)


def test_epoch(vae, device, dataloader):
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            encoded_data = vae.encoder(x)
            x_hat = vae(x)
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    dir = "./Brain Tumor MRI Classfication/"
    save_path = "./model/brain_tumor_epoch30.pt"
    train_data, val_data, test_data, test_path = fileopen_splitdata_labeling(dir=dir)
    class_num = len(set(train_data["label"]))

    train_dataset = dataset(train_data)
    val_dataset = dataset(val_data)
    test_dataset = dataset(test_data)

    print("\n###ë°ì´í°ì ì ììë£###")
    train_batch = 32
    val_batch = 1
    test_batch = 1

    lr = 0.00001
    step_size = 30
    gamma = 0.01
    total_epoch = 1
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch)

    torch.manual_seed(0)

    d = 4
    vae = VariationalAutoencoder(latent_dims=d)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    vae.to(device)

    signal = input(str("if you want to train press y or n : "))

    if signal == "y":
        lr = 1e-3
        num_epochs = 50
        optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(num_epochs):
            train_loss = train_epoch(vae, device, train_dataloader, optim)
            val_loss = test_epoch(vae, device, val_dataloader)
            print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
                                                                                  val_loss))
        torch.save(vae.state_dict(), save_path)

    if signal == "n":
        with torch.no_grad():
            vae.load_state_dict(torch.load(save_path))

            latent = torch.randn(128, d, device=device)

            img_recon = vae.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(20, 8.5))
            show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
            plt.show()