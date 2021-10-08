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

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tensorboard
    LOG_DIR = 'logs/pytorch'
    writer = SummaryWriter(log_dir=LOG_DIR)

    epochs = 5
    img_size = 50
    img_data, metadata, y = load_data(img_size=img_size)

    # Cross validation
    k_folds = 4
    cv_results = pd.DataFrame(columns=[f'fold_{i}' for i in range(k_folds)])
    kf = KFold(n_splits=k_folds)
    for fold, (train_index, validation_index) in enumerate(kf.split(y)):
        print('Fold {fold}')
        # Split data
        img_data_train, metadata_train, y_train = img_data[train_index], metadata[train_index], y[train_index]
        img_data_validation, metadata_validation, y_validation = img_data[validation_index], metadata[validation_index], y[validation_index]

        # Datasets
        training_dataset = PawpularityDataset(img_data_train, metadata_train, y_train)
        validation_dataset = PawpularityDataset(img_data_validation, metadata_validation, y_validation)

        # Dataloders
        training_dataloader = DataLoader(dataset=training_dataset, batch_size=8, shuffle=True, num_workers=2,
                                     prefetch_factor=2)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=8, shuffle=True, num_workers=2,
                                       prefetch_factor=2)

        # Model                     # tensorboard --logdir "Google Landmark Recognition 2021\logs"
        model = effnet_model().to(device)
        learning_rate = 1e-3

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        fold_results = []
        for epoch in range(epochs):
            print(f"-----------------Epoch {epoch + 1}-----------------")
            pytorch_train_loop(training_dataloader, y_train.shape[0], model, loss_fn, optimizer, writer, epoch, device)
            test_rmse = pytorch_test_loop(validation_dataloader, y_validation.shape[0], model, loss_fn, writer, epoch, device)
            fold_results.append(test_rmse)
        cv_results[f'fold_{fold}'] = fold_results
    cv_results.index = [i for i in range(cv_results.shape[0])]

    # Plot cross validation results per epoch
    print(cv_results)
    sns.set()
    fig = plt.figure(1)
    for i in range(k_folds):
        sns.lineplot(x=cv_results.index, y=f'fold_{i}', data=cv_results)
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Cross validation results')
    plt.legend(cv_results.columns)
    plt.savefig('results/cv_results.png')
    plt.show()

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
