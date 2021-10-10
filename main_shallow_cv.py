"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""

from utilities import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
import pandas as pd
from sklearn.metrics import r2_score
import time

if __name__ == '__main__':
    start = time.perf_counter()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 50
    img_data, metadata, y = load_data(img_size=img_size)
    metadata = metadata[:50]  # TODO remove debugging
    X = (img_data, metadata)
    y = y[:50]

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
    model = sklearn_wrapper(head=regressor, device=device)
    model.fit(X, y)

    pred = model.predict(X)
    r2 = r2_score(y_true=y, y_pred=pred)  # TODO remove debugging

    # Hyperparameter optimisation
    objective = define_objective(DecisionTreeRegressor(random_state=0), img_data=img_data, metadata=metadata, y=y,
                                 kfolds=5, device=device)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=20)
    print(study.best_params)
    print(study.best_value)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
