"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""

from utilities import *
from sklearn.tree import DecisionTreeRegressor
from os import makedirs
import pandas as pd
from pickle import dump
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
    if debugging:
        print("Debugging!!!")
    print(f"Using device: {device}")
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/shallow' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    img_size = 125
    img_data, metadata, y = load_train_data(img_size=img_size, device=device)
    X = (img_data, metadata)


    # Regressor head
    regressor = DecisionTreeRegressor(random_state=0)
    # regressor = RandomForestregressor(random_state=0)
    # regressor = Baggingregressor(base_estimator=DecisionTreeregressor(random_state=0), random_state=0)
    # regressor = LogisticRegression(random_state=0)
    # regressor = SVC(random_state=0)
    # regressor = KNeighborsregressor()
    # regressor = MLPregressor(random_state=0)
    # regressor = GaussianNB()
    # regressor = AdaBoostregressor(base_estimator=DecisionTreeregressor(random_state=0),
    #                                 random_state=0)
    # regressor = GradientBoostingregressor(random_state=0)

    # Model
    model = SKlearnWrapper(head=regressor, device=device)
    model.fit(X, y)

    # Hyperparameter optimisation
    study_name = f'cnn_study_{time_stamp}'
    objective = define_objective(DecisionTreeRegressor(random_state=0), img_data=img_data, metadata=metadata, y=y,
                                 kfolds=5, device=device)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name,
                                direction='minimize', storage=f'sqlite:///{LOG_DIR}{study_name}.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=15, timeout=25)
    print(study.best_params)
    print(study.best_value)

    # Save results
    results_dict = {'Best_hyperparameters': study.best_params, 'Best_value': study.best_value, 'study_name': study_name,
                    'log_dir': LOG_DIR}
    save_dict_to_file(results_dict, LOG_DIR, txt_name='study_results')
    df = study.trials_dataframe()
    df.to_csv(LOG_DIR + "study_results.csv")

    # Plot study results
    plots = [(optuna.visualization.plot_optimization_history, "optimization_history.png"),
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
