# flat-price-estimation

## Table of Contents

- [About](#about)
- [Usage](#usage)

## About
The aim of this project is to leverage the power of machine learning techniques in capturing complex relationships to develop a robust and accurate model for predicting flat prices in Indian cities, utilizing a dataset comprising entries with 9 key factors, including construction status, regulatory approvals (RERA), room count (BHK_NO.), square footage (SQUARE_FT), and location details (ADDRESS, LONGITUDE, LATITUDE). We attempt identify trends, correlations, and insights in the real estate market that are essential for comprehending how these factors affect flat pricing.

## Usage
This repository contains two code files, 'preprocess.py' and 'training_code.py' which contain the code for preprocessing the data and training different regression models on the preprocessed data respectively. After the models are trained on data with principal component analysis applied and without and the best hyperparameters for each model are determined, the predictions on the test data are obtained. The results from training are stored in the results folder, and the best scores obtained with each model using the r2 score and negative mean squared error as metrics, in the format 'results_metric_pca-usage.txt'. Similarly the best hyperparameters for each model are stored in JSON files in the format 'model-name_best_parameters_metric_pca-usage.json' and the predictions on the test data are stored in similarly named .npy files.
