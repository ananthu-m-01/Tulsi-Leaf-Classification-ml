# Tulsi Leaf Classification

This project aims to classify Rama Tulsi leaves into **Good** or **Bad** categories based on a dataset of images. The classification is done using machine learning techniques and image processing.

## Overview

Tulsi (Ocimum sanctum) is a sacred plant in Hinduism, known for its medicinal properties. The leaves of Tulsi plants are used for various purposes, and in this project, we aim to classify the leaves as **Good** or **Bad** based on their appearance.

### Types of Tulsi Leaves:
1. **Good Leaves**: Healthy, fresh, and free from diseases or damage.
2. **Bad Leaves**: Leaves that are damaged, infected, or have defects.

## Algorithm Used

The algorithm used in this project is **K-Nearest Neighbors (KNN)**. KNN is a simple, supervised machine learning algorithm used for classification tasks. It works by finding the closest labeled data points (neighbors) to the input data and predicting the class based on majority voting.

### KNN in this Project:
- **Training Data**: The images of both **Good** and **Bad** leaves are used to train the KNN model.
- **Feature Extraction**: Images are resized to 128x128 pixels and then flattened into a 1D array of pixel values.
- **Classification**: The model classifies each test image into either **Good** or **Bad** category based on the neighbors' labels.

## Project Structure

The project contains the following key files:

- **Tulsi_Leaf_Classification.ipynb**: The main Jupyter notebook containing the code for data loading, training, prediction, and result visualization.
- **output.png**: Output of the confusion matrix visualization.

## Working Flow

1. **Data Loading**:
   - The dataset containing **Good** and **Bad** leaf images is loaded using the `load_dataset` function.
   - The images are resized to 128x128 pixels and flattened for training.

2. **Data Preprocessing**:
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

3. **Model Training**:
   - The KNN model is trained using the training set (`X_train`, `y_train`).

4. **Prediction**:
   - The model is used to predict the class of test images (`X_test`).

5. **Result Visualization**:
   - A **confusion matrix** is displayed to show the classification performance.
   - The **classification report** gives detailed metrics like precision, recall, and F1-score.
   - A few test images are displayed with the predicted and actual labels.

6. **Real-Time Prediction**:
   - A custom test image can be classified by providing its path to the `predict_and_display_image` function.

7. **Execution Time**:
   - The execution time of the entire process is recorded and displayed.

## Requirements

- Python 3.x
- Libraries:
  - `opencv-python`
  - `numpy`
  - `sklearn`
  - `matplotlib`

To install the required libraries, run the following command:

```bash
pip install opencv-python numpy scikit-learn matplotlib
