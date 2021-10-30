"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""
import math

import pandas as pd
import numpy as np
import wandb
import cv2
import timm
from tqdm import tqdm
import optuna
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from statistics import mean
from random import sample
from copy import deepcopy
from math import sqrt


def define_objective(regressor, img_data, metadata, y, kfolds, device):
    def objective(trial):
        hyperparameters = {}
        model_name = str(regressor)

        if 'DecisionTree' in model_name:
            model_name = 'DecisionTree'
            hyperparameters['max_depth'] = trial.suggest_int('max_depth', 1, 50)
            hyperparameters['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
            hyperparameters['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            hyperparameters['splitter'] = trial.suggest_categorical('splitter', ["random", "best"])
            hyperparameters['max_features'] = trial.suggest_categorical('max_features', ["auto", "sqrt"])

        name = (model_name + str(hyperparameters)).replace(' ', '')
        print(f"model name = {model_name}")
        wandb.init(project="pawpularity-shallow", entity="nickkaparinos", name=name, config=hyperparameters,
                   reinit=True, group=model_name)
        regressor.set_params(**hyperparameters)
        model = SKlearnWrapper(head=regressor, device=device)

        k_folds = kfolds
        kf = KFold(n_splits=k_folds)

        cv_results = Parallel(n_jobs=kfolds, prefer="processes")(
            delayed(score)((img_data[train_index], metadata[train_index]), y[train_index],
                           (img_data[validation_index], metadata[validation_index]), y[validation_index],
                           deepcopy(model)) for train_index, validation_index in kf.split(y))
        val_rmse_list = [i[0] for i in cv_results]
        average_rmse = mean(val_rmse_list)
        val_r2_list = [i[1] for i in cv_results]
        wandb.log(data={'Mean Validation RMSE': average_rmse, 'Folds Validation RMSE': val_rmse_list,
                        'Mean Validation R2': mean(val_r2_list), 'Fold Validation R2': val_r2_list})
        return average_rmse

    return objective


def define_objective_neural_net(img_data, metadata, y, k_folds, epochs, model_type, notes, device):
    def objective(trial):
        kf = KFold(n_splits=k_folds)
        loss_fn = torch.nn.MSELoss()
        training_dataloaders = []
        validation_dataloaders = []
        optimizers = []
        learning_rate = 1e-3
        batch_size = 16

        # Models
        model_list, name, hyperparameters = create_models(model_type=model_type, trial=trial, k_folds=k_folds,
                                                          device=device)
        # config = hyperparameters | {'img_size': img_data.shape[1], 'epochs': epochs, 'learning_rate': learning_rate,
        #                             'batch_size': batch_size}
        config = dict(hyperparameters,
                      **{'img_size': img_data.shape[1], 'epochs': epochs, 'learning_rate': learning_rate,
                         'batch_size': batch_size})
        wandb.init(project=f"pawpularity-{model_type}", entity="nickkaparinos", name=name, config=config, notes=notes,
                   group=model_type, reinit=True)

        for fold, (train_index, validation_index) in enumerate(kf.split(y)):
            # Datasets
            training_dataset = PawpularityDataset(train_index)
            validation_dataset = PawpularityDataset(validation_index)

            # Dataloders
            training_dataloaders.append(
                DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                           prefetch_factor=1))
            validation_dataloaders.append(
                DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                           prefetch_factor=1))
            optimizers.append(torch.optim.Adam(model_list[fold].parameters(), lr=learning_rate))

        for epoch in tqdm(range(epochs)):
            train_rmse_list = []
            train_r2_list = []
            val_rmse_list = []
            val_r2_list = []
            for fold in range(k_folds):
                train_rmse, train_r2 = pytorch_train_loop(img_data, metadata, y, training_dataloaders[fold],
                                                          model_list[fold], loss_fn, optimizers[fold], epoch, device)
                val_rmse, val_r2 = pytorch_test_loop(img_data, metadata, y, validation_dataloaders[fold],
                                                     model_list[fold], loss_fn, epoch,
                                                     device)
                val_rmse_list.append(val_rmse)
                val_r2_list.append(val_r2)
                train_rmse_list.append(train_rmse)
                train_r2_list.append(train_r2)
            val_average_rmse = mean(val_rmse_list)
            wandb.log(data={'Epoch': epoch, 'Mean Training RMSE': mean(train_rmse_list),
                            'Mean Training R2': mean(train_r2_list),
                            'Mean Validation RMSE': val_average_rmse, 'Folds Validation RMSE': val_rmse_list,
                            'Mean Validation R2': mean(val_r2_list), 'Fold Validation R2': val_r2_list})
            # Pruning
            trial.report(val_average_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val_average_rmse

    return objective


class SKlearnWrapper():
    def __init__(self, head, device, input_channels=3, print_shape=False):
        # Use efficientnet backbone
        self.model = EfficientNet.from_pretrained('efficientnet-b3').to(device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.head = head
        self.device = device

    def fit(self, X, y):
        images, metadata = torch.from_numpy(X[0]), torch.from_numpy(X[1])
        images = images.permute(0, 3, 1, 2).to(
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
        images = images.permute(0, 3, 1, 2).to(
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


class SwinOptunaHypermodel(nn.Module):
    def __init__(self, n_linear_layers, n_neurons, p, input_channels=3):
        super().__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin.patch_embed = timm.models.layers.patch_embed.PatchEmbed(patch_size=4, embed_dim=128,
                                                                          norm_layer=nn.LayerNorm)
        self.fc1 = nn.LazyLinear(n_neurons)
        self.dropout = nn.Dropout(p=p)
        self.temp_layers = []
        for _ in range(n_linear_layers):
            self.temp_layers.append(nn.Linear(n_neurons, n_neurons))
        self.linear_layers = nn.ModuleList(self.temp_layers)
        self.output_layer = nn.Linear(n_neurons, 1)

    def forward(self, x):
        images, metadata = x
        x = self.swin(images)
        x = nn.Flatten()(x)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = nn.ReLU()(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


class EffnetOptunaHypermodel(nn.Module):
    def __init__(self, n_linear_layers, n_neurons, p, input_channels=3):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.LazyLinear(n_neurons)
        self.temp_layers = []
        self.dropout = nn.Dropout(p=p)
        for _ in range(n_linear_layers):
            self.temp_layers.append(nn.Linear(n_neurons, n_neurons))
        self.linear_layers = nn.ModuleList(self.temp_layers)
        self.output_layer = nn.Linear(n_neurons, 1)

    def forward(self, x):
        images, metadata = x
        x = self.model.extract_features(images)
        x = nn.Flatten()(x)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = nn.ReLU()(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


class EffnetModel(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # Use efficientnet
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.LazyLinear(256)
        # self.fc1 = nn.Linear(1292, 256)
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


class PawpularityDataset_OLD(torch.utils.data.Dataset):
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


class PawpularityDataset(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]


def load_train_data(img_size=256, device='cpu') -> tuple:
    """ Returns training set as list of (x,y) tuples
    where x = (resized_image, metadata)
    """
    train_metadata = pd.read_csv('train.csv')
    img_ids = train_metadata['Id']
    n_debug_images = 50
    # img_data = np.zeros((img_ids.shape[0], img_size, img_size, 3))  # TODO remove debugging img_ids.shape[0]
    img_data = torch.zeros((img_ids.shape[0], img_size, img_size, 3), dtype=torch.float32)
    metadata = train_metadata.iloc[:, 1:-1].values
    y = train_metadata.iloc[:, -1].values

    for idx, img_id in enumerate(tqdm(img_ids)):
        # if idx >= n_debug_images:  # TODO remove debugging
        #     break
        img_array = cv2.imread(f'train/{img_id}.jpg')
        img_array = cv2.resize(img_array, (img_size, img_size)) / 255
        img_data[idx, :, :, :] = torch.tensor(img_array)
    img_data = img_data.to(device)
    metadata = torch.tensor(metadata.astype(np.single)).to(device)
    y = torch.tensor(y.astype(np.single)).to(device)

    return img_data, metadata, y


def load_test_data(img_size=256) -> tuple:
    """ Returns test set as list of (x,y) tuples
    where x = (resized_image, metadata)
    """
    test_metadata = pd.read_csv('test.csv')
    img_ids = test_metadata['Id']
    img_data = np.zeros((img_ids.shape[0], img_size, img_size, 3))
    metadata = test_metadata.iloc[:, 1:].values

    for idx, img_id in enumerate(tqdm(img_ids)):
        img_array = cv2.imread(f'test/{img_id}.jpg')
        img_array = cv2.resize(img_array, (img_size, img_size)) / 255
        img_data[idx, :, :, :] = img_array

    return img_ids, img_data, metadata


def pytorch_train_loop(img_data, metadata, y, dataloader, model, loss_fn, optimizer, epoch, device) -> tuple:
    model.train()
    running_loss = 0.0
    y_list = []
    y_pred_list = []
    for batch, indices in enumerate(dataloader):
        img_data_batch = img_data[indices]
        metadata_batch = metadata[indices]
        y_batch = y[indices]
        img_data_batch = img_data_batch.permute(0, 3, 1, 2).to(
            device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)

        # Calculate loss function
        y_pred = model((img_data_batch, metadata_batch))
        loss = loss_fn(y_pred, y_batch.view(-1, 1))
        y_list.extend(y_batch.to('cpu').tolist())
        y_pred_list.extend(y_pred[:, 0].to('cpu').tolist())

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:
            wandb.log(data={'Epoch': epoch, 'Training_loss': running_loss / 100})
            running_loss = 0.0

    # Calculate and save metrics
    train_rmse = np.sqrt(mean_squared_error(y_list, y_pred_list))
    train_r2 = r2_score(y_list, y_pred_list)
    return train_rmse, train_r2


def pytorch_test_loop(img_data, metadata, y, dataloader, model, loss_fn, epoch, device) -> tuple:
    model.eval()
    running_loss = 0.0
    y_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch, indices in enumerate(dataloader):
            img_data_batch = img_data[indices]
            metadata_batch = metadata[indices]
            y_batch = y[indices]
            img_data_batch = img_data_batch.permute(0, 3, 1, 2).to(
                device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS) To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)

            # Calculate loss function
            y_pred = model((img_data_batch, metadata_batch))
            loss = loss_fn(y_pred, y_batch.view(-1, 1))
            y_list.extend(y_batch.to('cpu').tolist())
            y_pred_list.extend(y_pred[:, 0].to('cpu').tolist())

            running_loss += loss.item()
            if batch % 100 == 0:
                wandb.log(data={'Epoch': epoch, 'Validation_loss': running_loss / 100})
                running_loss = 0.0

    # Calculate and save metrics
    val_rmse = np.sqrt(mean_squared_error(y_list, y_pred_list))
    val_r2 = r2_score(y_list, y_pred_list)
    return val_rmse, val_r2


def create_models(model_type, trial, k_folds, device):
    """ Create and return a model list """
    if model_type == 'cnn':
        n_linear_layers = trial.suggest_int('n_linear_layers', 0, 4)
        n_neurons = trial.suggest_int('n_neurons', low=32, high=512, step=32)
        p = trial.suggest_float('dropout_p', low=0, high=0.5, step=0.1)
        model_list = [EffnetOptunaHypermodel(n_linear_layers=n_linear_layers, n_neurons=n_neurons, p=p).to(device) for _
                      in
                      range(k_folds)]
        name = f'{model_type}_neurons{n_neurons},layers{n_linear_layers},drop{p}'
        hyperparamers = {'n_neurons': n_neurons, 'n_linear_layers': n_linear_layers, 'dropout_p': p}
        return model_list, name, hyperparamers
    elif model_type == 'swin':
        n_linear_layers = trial.suggest_int('n_linear_layers', 0, 4)
        n_neurons = trial.suggest_int('n_neurons', low=32, high=512, step=32)
        p = trial.suggest_float('dropout_p', low=0, high=0.5, step=0.1)
        model_list = [SwinOptunaHypermodel(n_linear_layers=n_linear_layers, n_neurons=n_neurons, p=p).to(device) for _
                      in
                      range(k_folds)]
        name = f'{model_type}_neurons{n_neurons},layers{n_linear_layers},drop{p}'
        hyperparamers = {'n_neurons': n_neurons, 'n_linear_layers': n_linear_layers, 'dropout_p': p}
        return model_list, name, hyperparamers
    else:
        raise ValueError(f"Model type {model_type} not supported!")


def score(X_train, y_train, X_validation, y_validation, model) -> float:
    # Training
    model.fit(X_train, y_train)

    # Inference
    y_pred = model.predict(X_validation)
    val_rmse = sqrt(mean_squared_error(y_true=y_validation, y_pred=y_pred))
    val_r2 = r2_score(y_validation, y_pred)
    return val_rmse, val_r2


def save_dict_to_file(dict, path, txt_name='hyperparameter_dict'):
    f = open(path + '/' + txt_name + '.txt', 'w')
    f.write(str(dict))
    f.close()
