"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""

from utilities import *
from os import makedirs
import pandas as pd
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import time


if __name__ == '__main__':
    start = time.perf_counter()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")
    makedirs('logs/grad_cam', exist_ok=True)

    img_size = 400
    img_data, metadata, y = load_train_data(img_size=img_size)

    # Efficient net model
    model = EfficientNet.from_pretrained('efficientnet-b3').to(device=device)
    n_images = 10
    n_layer = -1

    for i in range(n_images):
        images = torch.from_numpy(img_data)
        images = images.permute(0, 3, 1, 2)
        test_image = images[None, i]
        target_layers = [model._blocks[n_layer]]

        # Grad Cam
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=test_image, target_category=None)
        grayscale_cam = grayscale_cam[0, :]

        # Visualize
        test_image = test_image[0].permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=False)

        # cv2.imshow(f'visualization_{i}', visualization)
        # cv2.waitKey(0)

        cv2.imwrite('logs/grad_cam/'+f'image_{i}_plusplus.png', visualization)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
