import numpy as np
from sklearn.manifold import TSNE
import torch
from PIL import Image, ImageDraw, ImageFont
import pickle

from model_struct_define.my_manifold_mixed_SE_resnet import se_resnet18
from torchvision.models import resnet18

import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from scipy.stats import entropy
from math import log2


root_dir = '/home/hail09/Documents/hail_moon/ImageNet10_experiments'
cache_dir = os.path.join(root_dir, 'Imagenette')
dataset_name = "frgfm/imagenette"
split_name = 'full_size'

if not os.path.exists(os.path.join(cache_dir, 'downloads')):
    dataset = load_dataset(dataset_name, split_name, cache_dir=cache_dir)
else:
    dataset = load_dataset(dataset_name, split_name, cache_dir=cache_dir, download_mode='reuse_cache_if_exists')

train_dataset = dataset['train']
train_all_labels = train_dataset['label']
unique_labels = set(train_all_labels)  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
train_unique_labels = list(unique_labels)[:9]

# create lists of images by label
train_img_list = {f"train_label{i}": [] for i in range(10)}

for train_data in train_dataset:
    image = train_data['image']
    label = train_data['label']
    train_img_list[f"train_label{label}"].append(image)

# preprocessing
# divide into major classes, moderate classes, rare classes
train_major_classes = []
train_moderate_classes = []
train_rare_classes = []

train_img_list["train_label0"] = train_img_list["train_label0"][:950]
train_img_list["train_label1"] = train_img_list["train_label1"][:450]
train_img_list["train_label2"] = train_img_list["train_label2"][:950]
train_img_list["train_label3"] = train_img_list["train_label3"][:450]
train_img_list["train_label4"] = train_img_list["train_label4"][:100]
train_img_list["train_label5"] = train_img_list["train_label5"][:450]
train_img_list["train_label6"] = train_img_list["train_label6"][:950]
train_img_list["train_label7"] = train_img_list["train_label7"][:100]
train_img_list["train_label8"] = train_img_list["train_label8"][:100]
train_img_list.pop("train_label9")

for key, img_list in train_img_list.items():
    if key == 'train_label0' or key == 'train_label2' or key == 'train_label6':
        train_major_classes.append((key, img_list))
    elif key == 'train_label1' or key == 'train_label5' or key == 'train_label9':
        train_moderate_classes.append((key, img_list))
    elif key == 'train_label8' or key == 'train_label4' or key == 'train_label7':
    # elif key in [4, 7, 8]:
        train_rare_classes.append((key, img_list))


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_ = self.images[idx]
        if img_.mode == 'L':
            img_ = img_.convert('RGB')
        img = self.transform(img_)
        label = self.labels[idx]
        return img, label


class_ratio = 0.2
class_image_counts = [950, 450, 950, 450, 100, 450, 950, 100, 100]
val_image_counts = [round(count * class_ratio) for count in class_image_counts]

train_images = []
train_labels = []
val_images = []
val_labels = []
for label, img_list in train_img_list.items():
    class_label = int(label.split('_')[-1][-1])
    num_images = len(img_list)
    num_val_images = val_image_counts[class_label]
    val_indices = np.random.choice(num_images, num_val_images, replace=False)
    train_images.extend([img for i, img in enumerate(img_list) if i not in val_indices])

    train_labels.extend([class_label] * (len(img_list) - num_val_images))
    val_images.extend([img_list[i] for i in val_indices])
    val_labels.extend([class_label] * num_val_images)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 32
train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


def extract_mlp_outputs(basic_resnet_flag, model, dataloader):
    all_outputs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            if basic_resnet_flag == 1:
                hiddens = model(images)
                all_outputs.append(hiddens.detach().cpu())
            else:
                features, hiddens = model(images, flag_extract=True)
                all_outputs.append(hiddens[0].detach().cpu())  # .numpy()붙여야할수도

    return np.concatenate(all_outputs, axis=0)


def calculate_kld(p, q):
    assert p.shape == q.shape, "Distributions must have the same shape"
    kld_rst = np.sum(p * np.log((p+(1e-5) )/ (q+(1e-5))))

    return kld_rst

root_dir = '/home/hail09/Documents/hail_moon/ImageNet10_experiments'
saved_model_dir = os.path.join(root_dir, 'saved_model/')
tsne_pkl_save_dir = os.path.join(root_dir, 'tSNE&KLD/tsne_pkl')
tsne_image_save_dir = os.path.join(root_dir, 'tSNE&KLD/tsne_image')

model_files = os.listdir(saved_model_dir)  # list
model_files.sort()
# model_files_sorted = sorted(model_files)

for model_file in model_files:
    if model_file.endswith(".pt"):
        model_path = os.path.join(saved_model_dir, model_file)

        pt_f_name = model_file.split('.')[0]
        pt_f_name_list = pt_f_name.split("_")
        tsne_f_name = "tsne_" + "_".join(pt_f_name_list[:4]) + ".jpg"
        tsne_img_save_path = os.path.join(tsne_image_save_dir, tsne_f_name)

        is_basic_resnet = 1
        if pt_f_name_list[0] == 'basic':  # basic resnet
            loaded_model = resnet18(pretrained=False)
            num_features = loaded_model.fc.in_features
            loaded_model.fc = nn.Linear(num_features, 9)  # Assuming 10 classes for our dataset
        elif pt_f_name_list[2] == 'resnet':  # manifold mixed resnet
            loaded_model = se_resnet18(9, if_mixup=True, if_se=False, if_shake_shake=False)
            is_basic_resnet = 0
        elif pt_f_name_list[2] == 'SE':  # manifold mixed SE resnet
            loaded_model = se_resnet18(9, if_mixup=True, if_se=True, if_shake_shake=False)
            is_basic_resnet = 0
        else:  # SE resnet
            loaded_model = se_resnet18(9, if_mixup=False, if_se=True, if_shake_shake=False)
            is_basic_resnet = 0

        loaded_model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 순서 중요!
        loaded_model = loaded_model.to(device)

        loaded_model.eval()

        mlp_outputs_rst = extract_mlp_outputs(is_basic_resnet, loaded_model, train_loader)

        # tsne here
        tsne = TSNE(n_components=2, random_state=42)
        outputs_embedded = tsne.fit_transform(mlp_outputs_rst)  # type: ndarray: (3600, 2)

# tSNE =================================================================================================================
        max_x_index = np.argmax(outputs_embedded[:, 0])
        max_y_index = np.argmax(outputs_embedded[:, 1])
        min_x_index = np.argmin(outputs_embedded[:, 0])
        min_y_index = np.argmin(outputs_embedded[:, 1])

        max_x_element = outputs_embedded[max_x_index]
        max_y_element = outputs_embedded[max_y_index]
        min_x_element = outputs_embedded[min_x_index]
        min_y_element = outputs_embedded[min_y_index]


        # save embeddings
        pkl_f_name = tsne_f_name.split('.')[0] + '.pkl'
        embeddings_save_path = os.path.join(tsne_pkl_save_dir, pkl_f_name)
        with open(embeddings_save_path, 'wb') as f:
            pickle.dump(outputs_embedded, f)

        class_colors = {
            0: (255, 0, 0),  # Red
            1: (0, 255, 0),  # Green
            2: (0, 0, 255),  # Blue
            3: (255, 255, 0),  # Yellow
            4: (255, 0, 255),  # Magenta
            5: (0, 255, 255),  # Cyan
            6: (128, 0, 0),  # Maroon
            7: (0, 128, 0),  # Green (Dark)
            8: (0, 0, 128),  # Navy
        }

        # visualize here
        width = 200
        height = 200
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        legend_font = ImageFont.load_default()

        legend_x, legend_y = 10, 10
        legend_spacing = 20
        for i, label in enumerate(train_unique_labels):
            legend_color = class_colors[label]
            legend_text = f"Class {label}"
            draw.text((legend_x, legend_y + i * legend_spacing), legend_text, font=legend_font, fill=legend_color)

        # Visualization loop
        for point, label in zip(outputs_embedded, train_labels):  # outputs_embedded & train_labels num: 3600
            x, y = point
            x = (x + 100) * (width // 200)
            y = (y + 100) * (height // 200)
            x, y = int(x), int(y)
            color = class_colors[label]
            img.putpixel((x, y), color)

        img.save(tsne_img_save_path)

# histogram & KL-Divergence ============================================================================================
        # 760*3 + 360 * 3 + 80 * 3 = 3600
        match_label_points_list = [[] for _ in range(9)]
        for point, label in zip(outputs_embedded, train_labels):
            match_label_points_list[label].append(point)

        class_kld = {4: [], 7: [], 8: []}
        class_histograms = {}
        hist_flats = []
        for label, points in enumerate(match_label_points_list):
            len_points = len(points)
            np_arr_points = np.array(points)
            x_values = np_arr_points[:, 0]
            y_values = np_arr_points[:, 1]
            now_hist, _, _ = np.histogram2d(x_values, y_values, bins=(np.arange(-47, 58, 10), np.arange(-63, 58, 10)))

            class_histograms[label] = now_hist
            hist_flat = now_hist.ravel() / len_points
            hist_flats.append(hist_flat)

        rare_labels = [4, 7, 8]
        not_rare_labels = [0, 1, 2, 3, 5, 6]
        for rare in rare_labels:
            for i in not_rare_labels:
                kld = calculate_kld(hist_flats[rare], hist_flats[i])
                class_kld[rare].append(kld)

        # Save class KLD dictionary to a pickle file
        with open(os.path.join(tsne_pkl_save_dir, 'class_kld.pkl'), 'wb') as f:
            pickle.dump(class_kld, f)
