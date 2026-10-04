# Hyperspectral Crop Classification – Indian Pines

## Overview

This project presents a hyperspectral image classification workflow using the Indian Pines dataset. A Streamlit web application allows users to select different feature sets and machine-learning models and evaluate their classification performance.

The application displays classification metrics, a confusion matrix, ground-truth map, and classified map.

## Dataset

The Indian Pines dataset is an AVIRIS hyperspectral image acquired over the Indian Pines test site in Indiana, USA.

- Image size: 145 × 145 pixels
- Spectral bands: 200
- Land-cover classes: 16

## Feature Selection and Dimensionality Reduction

The application provides the following feature options:

1. All bands
2. Evenly spaced bands
3. Mutual Information with redundancy penalty
4. Mutual Information top-N
5. Principal Component Analysis (PCA)

## Classification

The application allows the user to select a machine-learning classification model.

The current application includes SVM classification and the experimental results generated for the project.

## Evaluation Metrics

Classification performance is evaluated using:

- Overall Accuracy (OA)
- Kappa coefficient
- Macro F1-score
- Confusion matrix
- Classified map

## Result

For the SVM classifier using all 200 spectral bands:

- Overall Accuracy: 92.39%
- Kappa: 0.9131
- Macro F1-score: 0.916
- Features used: 200

## Streamlit Application

The application provides an interactive interface where the user can:

1. Select a feature set.
2. Select a classification model.
3. Run the classification.
4. View the classification metrics.
5. View the confusion matrix.
6. View the ground-truth map.
7. View the classified map.
