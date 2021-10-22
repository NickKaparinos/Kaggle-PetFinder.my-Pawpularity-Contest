"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""
import torch

from utilities import *
import time
import timm
import cv2
from pprint import pprint

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Timm and swi
    model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    all_swin_models = timm.list_models('*swin*')
    # pprint(all_swin_models)
    swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    swin.patch_embed = timm.models.layers.patch_embed.PatchEmbed(patch_size=4, embed_dim=128, norm_layer=nn.LayerNorm)
    swin = swin.to(device)

    epochs = 2
    k_folds = 4
    img_size = 224
    n_debug_images = 50
    img_data, metadata, y = load_data(img_size=img_size)
    metadata = metadata[:n_debug_images]  # TODO remove debugging
    X = (img_data, metadata)
    y = y[:n_debug_images]

    X_img = torch.Tensor(X[0]).to(device)
    X_img = X_img.permute(0, 3, 1, 2).to(device)

    fet = swin(X_img)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
