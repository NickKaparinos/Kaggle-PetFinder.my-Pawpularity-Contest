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
import time

if __name__ == '__main__':
    start = time.perf_counter()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/shallow' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    img_size = 50
    n_debug_images = 50
    img_data, metadata, y = load_train_data(img_size=img_size)
    metadata = metadata[:n_debug_images]  # TODO remove debugging
    X = (img_data, metadata)
    y = y[:n_debug_images]

    # Regressor head
    regressor = DecisionTreeRegressor(random_state=0)

    # Model
    model = SKlearnWrapper(head=regressor, device=device)
    model.fit(X, y)

    # Make submission
    img_ids, test_img_data, test_metadata = load_test_data(img_size=img_size)
    X = (test_img_data, test_metadata)
    y_pred = model.predict(X)
    submission = pd.DataFrame(columns=['Id','Pawpularity'])
    submission['Id'] = img_ids
    submission['Pawpularity'] = y_pred
    submission.to_csv('submission.csv',index=False)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
