import os
from datasets import load_dataset
from collections import Counter

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import wandb
import argparse

from model_struct_define.my_manifold_mixed_SE_resnet import se_resnet18

# parse =
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='manifold_mixed_resnet')
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()
# args =


wandb.init(project="Imagenette_data_imbalance", name=args.id, entity="camorineon", config=args.__dict__)

root_dir = '/home/hail09/Documents/hail_moon/ImageNet10_experiments'
save_model_dir = os.path.join(root_dir, 'saved_model')

cache_dir = os.path.join(root_dir, 'Imagenette')
dataset_name = "frgfm/imagenette"
split_name = 'full_size'

if not os.path.exists(os.path.join(cache_dir, 'downloads')):
    dataset = load_dataset(dataset_name, split_name, cache_dir=cache_dir)
else:
    dataset = load_dataset(dataset_name, split_name, cache_dir=cache_dir, download_mode='reuse_cache_if_exists')


print(dataset)  # train: 9469, validation: 3925

train_dataset = dataset['train']
test_dataset = dataset['validation']
train_all_labels = train_dataset['label']
test_all_labels = test_dataset['label']
unique_labels = set(train_all_labels)  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


# count labels num
train_label_counter = Counter(train_all_labels)
test_label_counter = Counter(test_all_labels)

train_each_label_num = []
test_each_label_num = []
for label, count in train_label_counter.items():
    train_each_label_num.append([label, count])

for label, count in test_label_counter.items():
    test_each_label_num.append([label, count])
#
# train_each_label_num.sort(key=lambda x: x[1], reverse=True)
# test_each_label_num.sort(key=lambda x: x[1], reverse=True)
#
for label, num in train_each_label_num:
    print(f"<train label> {label}: {num}")
print('\n')
for label, num in test_each_label_num:
    print(f"<test label> {label}: {num}")


# ## print files name
# flag = 0
# for data in train_dataset:
#     if flag == 3001:
#         break
#     image_path = str(data['image'])
#     image_file = os.path.basename(image_path)
#     label = data['label']
#     print(image_file)
#     print(label)
#     flag += 1


# create lists of images by label
train_img_list = {f"train_label{i}": [] for i in range(10)}
test_img_list = {f"test_label{i}": [] for i in range(10)}

for train_data in train_dataset:
    image = train_data['image']
    label = train_data['label']
    train_img_list[f"train_label{label}"].append(image)

for test_data in test_dataset:
    image = test_data['image']
    label = test_data['label']
    test_img_list[f"test_label{label}"].append(image)


# preprocessing
# divide into major classes, moderate classes, rare classes
train_major_classes = []
train_moderate_classes = []
train_rare_classes = []
test_major_classes = []
test_moderate_classes = []
test_rare_classes = []


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

for key, img_list in test_img_list.items():
    test_img_list[key] = img_list[:350]

for key, img_list in train_img_list.items():
    if key == 'train_label0' or key == 'train_label2' or key == 'train_label6':
        train_major_classes.append((key, img_list))
    elif key == 'train_label1' or key == 'train_label5' or key == 'train_label9':
        train_moderate_classes.append((key, img_list))
    elif key == 'train_label8' or key == 'train_label4' or key == 'train_label7':
    # elif key in [4, 7, 8]:
        train_rare_classes.append((key, img_list))

for key, img_list in test_img_list.items():
    if key == 'test_label0' or key == 'test_label2' or key == 'test_label6':
        test_major_classes.append((key, img_list))
    elif key == 'test_label1' or key == 'test_label5' or key == 'test_label9':
        test_moderate_classes.append((key, img_list))
    elif key == 'test_label8' or key == 'test_label4' or key == 'test_label7':
        test_rare_classes.append((key, img_list))

# ======================================================================================================================


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
save_every = 100

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

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)

# Define the DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the ResNet model
manifold_mixed_resnet_model = se_resnet18(9, if_mixup=True, if_se=False, if_shake_shake=False)

# Replace the last fully connected layer for 1000 classes with a new one for 10 classes
# num_features = resnet_model.fc.in_features
# resnet_model.fc = nn.Linear(num_features, 9)  # Assuming 10 classes for our dataset

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifold_mixed_resnet_model = manifold_mixed_resnet_model.to(device)


def test(model, test_data, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for label, images in test_data:
            label_number = int(label.split('_')[-1][-1])
            labels = torch.tensor([label_number] * len(images)).to(device)

            tmp = []
            for img in images:
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = val_transform(img)
                tmp.append(img)
            images = torch.stack(tmp)

            outputs = model(images.to('cuda'))
            _, predicted = torch.max(outputs, 1)  # [[0, 0, 0, 0.4, 0.8, 0.2, 0, 0, 0]] => 4  [100]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Define beta distribution
def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(manifold_mixed_resnet_model.parameters(), lr=args.lr)

num_epochs = 1000
for epoch in range(num_epochs):
    manifold_mixed_resnet_model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients
        # TODO: no mixup
        # outputs = manifold_mixed_resnet_model(images)  # Forward pass

        # TODO: mixup
        # if you use manifold mixup
        lam = mixup_data(alpha=1)
        lam = torch.from_numpy(np.array([lam]).astype('float32')).to(device)
        outputs, reweighted_target = manifold_mixed_resnet_model(images, lam=lam, target=labels)

        loss = criterion(outputs, reweighted_target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * images.size(0)

        _, predicted_train = torch.max(outputs, 1)
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)  # Accumulate total count

    train_acc = 100 * correct_train / total_train

    # Validation
    manifold_mixed_resnet_model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = manifold_mixed_resnet_model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item() * images.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            total_val += labels.size(0)  # Accumulate total count
            correct_val += (predicted == labels).sum().item()  # Accumulate correct predictions

    # Print statistics
    epoch_loss = running_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    val_acc = 100 * correct_val / total_val

    test_major_acc = test(manifold_mixed_resnet_model, test_major_classes, device)
    wandb.log({"Test Accuracy (Major Classes)": test_major_acc}, step=epoch)

    # Log test accuracy for moderate classes
    test_moderate_acc = test(manifold_mixed_resnet_model, test_moderate_classes, device)

    wandb.log({"Test Accuracy (Moderate Classes)": test_moderate_acc}, step=epoch)

    # Log test accuracy for rare classes
    test_rare_acc = test(manifold_mixed_resnet_model, test_rare_classes, device)
    wandb.log({"Test Accuracy (Rare Classes)": test_rare_acc}, step=epoch)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Acc: {train_acc:.2f}%, "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Acc: {val_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, "
          f"Test Major Acc: {test_major_acc:.2f}%, "
          f"Test Mod Acc: {test_moderate_acc:.2f}%, "
          f"Test Rare Acc: {test_rare_acc:.2f}%, ")

    # if (epoch + 1) == num_epochs:
    #     save_model_path = os.path.join(save_model_dir, "manifold_mixed_resnet_final_model.pt")
    #     torch.save(manifold_mixed_resnet_model.state_dict(), save_model_path)
    #     wandb.save(save_model_path)

    if (epoch + 1) == num_epochs:
        save_model_path = os.path.join(save_model_dir, "manifold_mixed_resnet_final_model.pt")
        torch.save(manifold_mixed_resnet_model.state_dict(), save_model_path)
        wandb.save(save_model_path)
    else:
        if (epoch + 1) % save_every == 0:
            save_model_path = os.path.join(save_model_dir, "manifold_mixed_resnet_e{}_model.pt".format(epoch + 1))
            torch.save(manifold_mixed_resnet_model.state_dict(), save_model_path)
print("Finished Training")
