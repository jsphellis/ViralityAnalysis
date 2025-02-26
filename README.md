# TikTok Virality Prediction

This repository contains a deep learning project that predicts the virality of TikTok videos using computer vision techniques. The model analyzes video content to classify videos as viral or non-viral and predict a continuous virality score.

## Project Overview

The project uses a 3D convolutional neural network (specifically R(2+1)D-18 pre-trained on Kinetics-400) to analyze TikTok video content and predict:
1. Binary classification: Whether a video will go viral (is_viral)
2. Regression: A continuous virality score based on engagement metrics

## Repository Structure

- `Training.py`: Contains the main training pipeline for both classification and regression models
- `Evaluation.py`: Evaluates trained models on test data and generates performance metrics
- `Preprocessing.py`: Processes raw video data and calculates virality scores based on engagement metrics

## Data Processing

The project uses a custom `VideoDataset` class that:
- Loads video files from the TikTok dataset
- Handles variable-length videos through padding or random sampling
- Applies transformations for model input

Virality scores are calculated using a weighted combination of engagement metrics:
- Likes (diggCount): 40%
- Views (playCount): 30%
- Shares (shareCount): 20%
- Comments (commentCount): 10%

## Models

The project uses a modified R(2+1)D-18 architecture with:
- Pre-trained weights from Kinetics-400 dataset
- Custom fully connected layers for classification/regression
- Dropout layers to prevent overfitting

## Evaluation Metrics

### Classification Model
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Visualization: ROC curves, confusion matrices, precision-recall curves

### Regression Model
- Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
- Analysis of best and worst predictions

## Results

The models demonstrate the ability to analyze video content for virality prediction. The classification model achieves moderate accuracy, while the regression model provides continuous virality score predictions.