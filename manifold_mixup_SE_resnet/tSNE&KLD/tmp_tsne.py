import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import torch
from PIL import Image

from train.train_manifold_mixed_resnet import val_loader
from model_struct_define.my_manifold_mixed_SE_resnet import se_resnet18


def extract_mlp_outputs(model, dataloader):
    all_outputs = []
    with torch.no_grad():
        for images, _ in dataloader:
            features, hiddens = model(images, flag_extract=True)
            # mlp_outputs = model.fc(features.view(features.size(0), -1)).detach().numpy()
            # all_outputs.append(mlp_outputs)
            all_outputs.append(hiddens[0])
    return np.concatenate(all_outputs, axis=0)


# def visualize_with_tsne_and_save(outputs, save_path):
#     tsne = TSNE(n_components=2, random_state=42)
#     outputs_embedded = tsne.fit_transform(outputs)
#
#     img = Image.new('RGB', (800, 600), color='white')
#
#     for point in outputs_embedded:
#         x, y = point
#         x = int((x + 20) * 20)  # Scale x-coordinate
#         y = int((y + 20) * 20)  # Scale y-coordinate
#         img.putpixel((x, y), (0, 0, 0))  # Set pixel color
#
#     img.save(save_path)


root_dir = '/home/hail09/Documents/hail_moon/ImageNet10_experiments'
save_model_dir = os.path.join(root_dir, 'saved_model')
model_path = os.path.join(root_dir, 'saved_model/manifold_mixed_resnet_final_model.pt')
tsne_save_dir = os.path.join(root_dir, 'tsne/tsne_images')

tsne_filename = os.path.basename(model_path)
tsne_filename = 'tsne_' + os.path.splitext(tsne_filename)[0].replace("_final_model", "") + '.jpg'
tsne_img_save_path = os.path.join(tsne_save_dir, tsne_filename)

loaded_model = se_resnet18(9, if_mixup=True, if_se=False, if_shake_shake=False)
loaded_model.load_state_dict(torch.load(model_path))

loaded_model.eval()

mlp_outputs_rst = extract_mlp_outputs(loaded_model, val_loader)

# tsne here
tsne = TSNE(n_components=2, random_state=42)
outputs_embedded = tsne.fit_transform(mlp_outputs_rst)

# save
import pickle

with open(file='tsne.pkl', mode='wb') as f:
    pickle.dump(outputs_embedded, f)

# visualize here
# visualize_with_tsne_and_save(mlp_outputs_rst, tsne_img_save_path)
img = Image.new('RGB', (800, 600), color='white')

for point in outputs_embedded:
    x, y = point
    x = int((x + 20) * 20)  # Scale x-coordinate
    y = int((y + 20) * 20)  # Scale y-coordinate
    img.putpixel((x, y), (0, 0, 0))  # Set pixel color

img.save(tsne_img_save_path)
