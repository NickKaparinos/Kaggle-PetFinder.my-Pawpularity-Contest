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
from os import makedirs
from pickle import dump
import cv2
from pprint import pprint
import logging
import sys

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")

    # Timm and swin transformer
    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    # all_swin_models = timm.list_models('*swin*')
    # pprint(all_swin_models)

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/swin' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    epochs = 2
    k_folds = 4
    img_size = 224
    n_debug_images = 5
    img_data, metadata, y = load_train_data(img_size=img_size)
    metadata = metadata[:n_debug_images]  # TODO remove debugging
    X = (img_data, metadata)
    y = y[:n_debug_images]

    # Hyperparameter optimisation
    study_name = f'swin_study_{time_stamp}'
    notes = 'optimizer:Adam, swin_base_patch4_window7_224'
    objective = define_objective_neural_net(img_data=img_data, metadata=metadata, y=y, k_folds=k_folds, epochs=epochs,
                                            model_type='swin', notes=notes, device=device)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name,
                                direction='minimize', pruner=optuna.pruners.HyperbandPruner(),
                                storage=f'sqlite:///{LOG_DIR}{study_name}.db', load_if_exists=True)
    study.optimize(objective, n_trials=None, timeout=20)
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best value: {study.best_value}')

    # Save results
    results_dict = {'Best_hyperparameters': study.best_params, 'Best_value': study.best_value, 'study_name': study_name,
                    'log_dir': LOG_DIR}
    save_dict_to_file(results_dict, LOG_DIR, txt_name='study_results')
    df = study.trials_dataframe()
    df.to_csv(LOG_DIR + "study_results.csv")

    # Plot study results
    plots = [(optuna.visualization.plot_optimization_history, "optimization_history.png"),
             (optuna.visualization.plot_intermediate_values, "intermediate_values.png"),
             (optuna.visualization.plot_parallel_coordinate, "parallel_coordinate.png"),
             (optuna.visualization.plot_contour, "contour.png"),
             (optuna.visualization.plot_param_importances, "param_importances.png")]
    figs = []
    for plot_function, plot_name in plots:
        fig = plot_function(study)
        figs.append(fig)
        fig.write_image(LOG_DIR + plot_name)
        # fig.show()
    with open(LOG_DIR + 'result_figures.pkl', 'wb') as f:
        dump(figs, f)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
