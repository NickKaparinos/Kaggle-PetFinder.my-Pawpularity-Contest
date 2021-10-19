"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""
import torch
from utilities import *
import time
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from os import makedirs
from os import makedirs
import logging
import sys

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tensorboard
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/cnn' + time_stamp
    makedirs(f'results/{time_stamp}/')
    writer = SummaryWriter(log_dir=LOG_DIR)

    epochs = 6
    k_folds = 4
    img_size = 40
    n_debug_images = 50
    img_data, metadata, y = load_data(img_size=img_size)
    metadata = metadata[:n_debug_images]  # TODO remove debugging
    X = (img_data, metadata)
    y = y[:n_debug_images]

    # Hyperparameter optimisation
    objective = define_objective_cnn(img_data=img_data, metadata=metadata, y=y, k_folds=k_folds, epochs=epochs,
                                     writer=writer, device=device)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=f'cnn_study_{time_stamp}', direction='minimize',
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=None, timeout=5*60)
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best value: {study.best_value}')

    # Plot study results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"results/{time_stamp}/optimization_history.png")
    fig.show()
    fig2 = optuna.visualization.plot_intermediate_values(study)
    fig2.write_image(f"results/{time_stamp}/intermediate_values.png")
    fig2.show()
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.write_image(f"results/{time_stamp}/parallel_coordinate.png")
    fig3.show()
    fig4 = optuna.visualization.plot_contour(study)
    fig4.write_image(f"results/{time_stamp}/contour.png")
    fig4.show()
    fig5 = optuna.visualization.plot_param_importances(study)
    fig5.write_image(f"results/{time_stamp}/param_importances.png")
    fig5.show()

    # Execution Time # tensorboard --logdir "Petfinder-Pawpularity\logs"
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
