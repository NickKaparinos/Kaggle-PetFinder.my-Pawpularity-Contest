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
    y = y[:50]
    X = [img_data, metadata]
    while len(X) != len(y):
        X.append(0)

    # Regressor head
    regressor = DecisionTreeRegressor(random_state=0)
    # regressor = RandomForestregressor(random_state=0)
    # regressor = Baggingregressor(base_estimator=DecisionTreeregressor(random_state=0), random_state=0)
    # # regressor = LogisticRegression(random_state=0)
    # regressor = SVC(random_state=0)
    # regressor = KNeighborsregressor()
    # regressor = MLPregressor(random_state=0)
    # regressor = GaussianNB()
    # regressor = AdaBoostregressor(base_estimator=DecisionTreeregressor(random_state=0),
    #                                 random_state=0)
    # regressor = GradientBoostingregressor(random_state=0)

    # Model
    model = sklearn_wrapper(head=regressor, device=device)

    # Hyper parameter grid
    param_grid = {}
    if 'LogisticRegression' in str(regressor):
        param_grid = {'clf__C': [0.1, 0.5, 1, 2, 5, 10]}

    if 'SVC' in str(regressor):
        param_grid = {'clf__kernel': ['rbf'], 'clf__C': [0.1, 0.5, 1, 2, 5, 10],
                      'clf__gamma': ['scale', 'auto', 0.1, 0.5, 1, 2, 5, 10]}

    if 'KNeighbors' in str(regressor):
        param_grid = {'clf__n_neighbors': [3, 5, 7, 9]}

    if 'DecisionTree' in str(regressor):
        # param_grid = {'clf__criterion': ['gini', 'entropy'], 'clf__min_samples_leaf': [1, 2, 3],
        #               'clf__max_depth': [1, 2, 3, 5, 7, 10, 15, None], 'clf__min_samples_split': [2, 3, 4, 5, 6],
        #               'clf__max_features': ['auto', 'sqrt', 'log2']}
        param_grid = {'max_depth': [5, None]}

    if 'RandomForest' in str(regressor):
        param_grid = {'clf__n_estimators': [200, 300, 500], 'clf__criterion': ['gini', 'entropy'],
                      'clf__min_samples_leaf': [1, 2, 3], 'clf__max_depth': [3, 5, 7, 10, 15, None],
                      'clf__min_samples_split': [2, 3, 4, 5], 'clf__max_features': ['auto', 'sqrt', 'log2']}

    if 'MLPregressor' in str(regressor):
        param_grid = {
            'clf__hidden_layer_sizes': [(10), (50), (100), (10, 10), (50, 50), (100, 100), (10, 10, 10), (50, 50, 50),
                                        (100, 100, 100)],
            'clf__activation': ['identity', 'relu'], 'clf__solver': ['lbfgs', 'adam'],
            'clf__alpha': [0.0001, 0.0005, 0.001],
            'clf__max_iter': [200, 400]}

    if 'Baggingregressor' in str(regressor):
        param_grid = {'clf__n_estimators': [25, 50, 100, 200],
                      'clf__max_features': [1.0, 0.8, 0.5],
                      'clf__max_samples': [1.0, 0.8, 0.5],
                      'clf__base_estimator__max_depth': [1, 2, 3, 5, 10, 15, None],
                      'clf__base_estimator__criterion': ['gini', 'entropy'],
                      'clf__base_estimator__min_samples_split': [2, 3]}

    if 'AdaBoost' in str(regressor):
        param_grid = {'clf__n_estimators': [10, 25, 50, 100, 200], 'clf__learning_rate': [0.5, 0.75, 1.0, 1.25, 1.5],
                      'clf__base_estimator__criterion': ['gini', 'entropy'],
                      'clf__base_estimator__min_samples_leaf': [1, 2, 3],
                      'clf__base_estimator__max_depth': [2, 3, 5, 10, None],
                      'clf__base_estimator__min_samples_split': [2, 3]}

    if 'GradientBoosting' in str(regressor):
        param_grid = {'clf__learning_rate': [0.25, 0.5, 1.0, 1.5], 'clf__n_estimators': [50, 100, 200],
                      'clf__subsample': [1.0, 0.75, 0.5], 'clf__max_depth': [2, 3, 5, None],
                      'clf__max_features': ['auto', 'sqrt'], 'clf__min_samples_leaf': [1, 2, 3],
                      'clf__min_samples_split': [2, 3, 4, 5]}

    # Cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=5, verbose=2, scoring='accuracy',
                        return_train_score=False, refit=True)
    grid.fit(X, y)

    # Results
    cv_results = pd.DataFrame(grid.cv_results_)
    print(cv_results)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
