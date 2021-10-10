"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""
import pandas as pd
import numpy as np
from os import makedirs
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import optuna


def define_objective(regressor, img_data, metadata, y, kfolds, device):
    def objective(trial):
        hyperparameters = {}

        if 'DecisionTree' in str(regressor):
            hyperparameters['max_depth'] = trial.suggest_int('max_depth', 1, 50)
            hyperparameters['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
            hyperparameters['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            hyperparameters['splitter'] = trial.suggest_categorical('splitter', ["random", "best"])
            hyperparameters['max_features'] = trial.suggest_categorical('max_features', ["auto", "sqrt"])

        regressor.set_params(**hyperparameters)
        model = sklearn_wrapper(head=regressor, device=device)

        k_folds = kfolds
        cv_results = []
        kf = KFold(n_splits=k_folds)
        for fold, (train_index, validation_index) in enumerate(kf.split(y)):
            # Split data
            X_train, y_train = (img_data[train_index], metadata[train_index]), y[train_index]
            X_validation, y_validation = (img_data[validation_index], metadata[validation_index]), y[validation_index]

            # Training
            model.fit(X_train, y_train)

            # Inference
            y_pred = model.predict(X_validation)
            rmse = sqrt(mean_squared_error(y_true=y_validation, y_pred=y_pred))
            cv_results.append(rmse)

        return sum(cv_results) / kfolds

    return objective


class sklearn_wrapper():
    def __init__(self, head, device, input_channels=3, print_shape=False):
        # Use efficientnet backbone
        self.model = EfficientNet.from_pretrained('efficientnet-b0').to(device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.head = head
        self.device = device

    def fit(self, X, y):
        images, metadata = torch.from_numpy(X[0]), torch.from_numpy(X[1])
        images = images.permute(0, 3, 2, 1).to(
            self.device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
        x = self.model.extract_features(images.float())
        x = nn.Flatten()(x)
        # print(f'x shape before: {x.shape}')
        # print(f'metadata shape before: {metadata.shape}')
        X = torch.cat((x.to('cpu'), metadata), dim=1)
        # print(f'X shape after: {X.shape}')
        X = X.numpy()
        self.head.fit(X, y)

    def predict(self, X):
        images, metadata = torch.from_numpy(X[0]), torch.from_numpy(X[1])
        images = images.permute(0, 3, 2, 1).to(
            self.device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
        x = self.model.extract_features(images.float())
        x = nn.Flatten()(x)
        X = torch.cat((x.to('cpu'), metadata), dim=1)
        X = X.numpy()
        temp = self.head.predict(X)
        return temp

    def get_params(self, deep=True):
        return self.head.get_params(deep=deep)

    def set_params(self, **params):
        self.head.set_params(**params)


class effnet_model(nn.Module):
    def __init__(self, input_channels=3, print_shape=False):
        super().__init__()
        # Use efficientnet
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(1292, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        images, metadata = x
        x = self.model.extract_features(images)
        x = nn.Flatten()(x)
        # print(f'x shape before: {x.shape}')
        # print(f'metadata shape before: {metadata.shape}')
        x = torch.cat((x, metadata), dim=1)
        # print(f'x shape after: {x.shape}')
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.output_layer(x)
        return x


class PawpularityDataset(torch.utils.data.Dataset):
    def __init__(self, images, metadata, y):
        self.images = torch.Tensor(images)
        self.metadata = torch.Tensor(metadata)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img_array = self.images[index]
        metadata = self.metadata[index]
        y = self.y[index]

        return img_array, metadata, y


def load_data(img_size=256) -> tuple:
    """ Returns training set as list of (x,y) tuples
    where x = (resized_image, metadata)
    """
    train_metadata = pd.read_csv('train.csv')
    img_ids = train_metadata['Id']
    img_data = np.zeros((50, img_size, img_size, 3))  # TODO remove debugging img_ids.shape[0]
    metadata = train_metadata.iloc[:, 1:-1].values
    y = train_metadata.iloc[:, -1].values

    for idx, img_id in enumerate(tqdm(img_ids)):
        if idx >= 50:  # TODO remove debugging
            break
        img_array = cv2.imread(f'train/{img_id}.jpg')
        img_array = cv2.resize(img_array, (img_size, img_size)) / 255
        img_data[idx, :, :, :] = img_array

    return img_data, metadata, y


def pytorch_train_loop(dataloader, size, model, loss_fn, optimizer, writer, epoch, device):
    running_loss = 0.0

    for batch, X in enumerate(tqdm(dataloader)):
        images, metadata, y = X
        metadata = metadata.to(device)
        y = y.to(device)
        images = images.permute(0, 3, 2, 1).to(
            device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)

        # Calculate loss function
        y_pred = model((images, metadata))
        loss = loss_fn(y_pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and save metrics
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            writer.add_scalar('training_loss', running_loss / 1000, epoch * len(dataloader) + batch)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    average_rmse = np.sqrt(running_loss / size)
    writer.add_scalar('RMSE loss', average_rmse, epoch + 1)
    print(f"Train Error:  RMSE: {average_rmse:>0.8f}%\n")


def pytorch_test_loop(dataloader, size, model, loss_fn, writer, epoch, device) -> float:
    test_loss = 0.0

    with torch.no_grad():
        for batch, X in enumerate(tqdm(dataloader)):
            images, metadata, y = X
            metadata = metadata.to(device)
            y = y.to(device)
            images = images.permute(0, 3, 2, 1).to(
                device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)

            # Calculate loss function
            y_pred = model((images, metadata))
            test_loss += loss_fn(y_pred, y).item()

    # Calculate and save metrics
    test_rmse = np.sqrt(test_loss)
    writer.add_scalar('test_rmse', test_rmse, epoch)
    print(f"Test Error: RMSE {(test_rmse):>0.8f}%\n")
    return test_rmse
