# Kaggle-PetFinder.my-Pawpularity-Contest
A picture is worth a thousand words. But did you know a picture can save a thousand lives? Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. You might expect pets with attractive photos to generate more interest and be adopted faster. But what makes a good picture? With the help of data science, you may be able to accurately determine a pet photo’s appeal and even suggest improvements to give these rescue animals a higher chance of loving homes.

PetFinder.my is Malaysia’s leading animal welfare platform, featuring over 180,000 animals with 54,000 happily adopted. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Currently, PetFinder.my uses a basic Cuteness Meter to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved.

In this competition, raw images and metadata are analyzed for every sample to predict the “Pawpularity” of pet photos.


## Dataset
The dataset consists of 9912 samples. Each sample corresponds to a unique pet and consists of an image and 12 metadata features.

## Approaches
Three approaches were followed. Each approach consists of different Neural Network architectures and makes use of both the image and the metadata for each sample, in order to achieve optimal performace. Each model was tuned using Optuna\`s Bayesian Hyperparameter tuning.


## Aproach 1: CNN Encoder + Shallow Reggresion Head
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-PetFinder.my-Pawpularity-Contest/blob/master/images/pawpularity_shallow.png" alt="drawing" width="1000"/>

## Aproach 2: CNN Encoder + MLP Head
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-PetFinder.my-Pawpularity-Contest/blob/master/images/pawpularity_mlp.png" alt="drawing" width="1000"/>

## Aproach 3: Swin Visual Transformer + MLP Head
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-PetFinder.my-Pawpularity-Contest/blob/master/images/pawpularity_swin.png" alt="drawing" width="1000"/>
