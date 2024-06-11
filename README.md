# ML Cheatsheet

Cheat sheet for ML models based on scikit-learn, TensorFlow, PyTorch, matplotlib, NumPy, and pandas.

## Introduction

This repository contains a collection of example machine learning source codes for various ML frameworks and libraries such as scikit-learn, TensorFlow, PyTorch, matplotlib, NumPy, and pandas. The purpose of this cheatsheet is to provide a quick reference for students and developers to understand and implement various machine learning models and techniques.

## Installation

To use the examples in this repository, you need to have Python installed on your machine. You can install the required libraries using pip:

```bash
pip install scikit-learn tensorflow torch matplotlib numpy pandas
```

To use the examples using torchvision and torchaudio, you can install them using pip:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Table of Contents

- [ML Cheatsheet](#ml-cheatsheet)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Table of Contents](#table-of-contents)
  - [scikit-learn](#scikit-learn)
    - [Classification](#classification)
    - [Regression](#regression)
    - [Clustering](#clustering)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Model Evaluation and Selection](#model-evaluation-and-selection)
  - [TensorFlow](#tensorflow)
    - [Basic Operations](#basic-operations)
    - [Neural Networks](#neural-networks)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
    - [Model Saving and Loading](#model-saving-and-loading)
  - [PyTorch](#pytorch)
    - [Basic Operations](#basic-operations-1)
    - [Neural Networks](#neural-networks-1)
    - [Model Training](#model-training-1)
    - [Model Evaluation](#model-evaluation-1)
    - [Model Saving and Loading](#model-saving-and-loading-1)
  - [matplotlib](#matplotlib)
    - [Basic Plots](#basic-plots)
    - [Advanced Plots](#advanced-plots)
    - [Customization](#customization)
  - [NumPy](#numpy)
    - [Array Operations](#array-operations)
    - [Mathematical Operations](#mathematical-operations)
    - [Linear Algebra](#linear-algebra)
    - [Random Sampling](#random-sampling)
  - [pandas](#pandas)
    - [DataFrame Operations](#dataframe-operations)
    - [Data Cleaning](#data-cleaning)
    - [Data Aggregation](#data-aggregation)
    - [Time Series Analysis](#time-series-analysis)

## scikit-learn

### Classification

- [`classification/logistic_regression.py`](scikit-learn/classification/logistic_regression.py)
- [`classification/decision_tree.py`](scikit-learn/classification/decision_tree.py)
- [`classification/random_forest.py`](scikit-learn/classification/random_forest.py)
- [`classification/support_vector_machine.py`](scikit-learn/classification/support_vector_machine.py)

### Regression

- [`regression/linear_regression.py`](scikit-learn/regression/linear_regression.py)
- [`regression/ridge_regression.py`](scikit-learn/regression/ridge_regression.py)
- [`regression/decision_tree_regression.py`](scikit-learn/regression/decision_tree_regression.py)
- [`regression/random_forest_regression.py`](scikit-learn/regression/random_forest_regression.py)

### Clustering

- [`clustering/k_means.py`](scikit-learn/clustering/k_means.py)
- [`clustering/hierarchical_clustering.py`](scikit-learn/clustering/hierarchical_clustering.py)
- [`clustering/dbscan.py`](scikit-learn/clustering/dbscan.py)

### Dimensionality Reduction

- [`dimensionality_reduction/pca.py`](scikit-learn/dimensionality_reduction/pca.py)
- [`dimensionality_reduction/lda.py`](scikit-learn/dimensionality_reduction/lda.py)

### Model Evaluation and Selection

- [`model_evaluation/cross_validation.py`](scikit-learn/model_evaluation/cross_validation.py)
- [`model_evaluation/grid_search.py`](scikit-learn/model_evaluation/grid_search.py)
- [`model_evaluation/random_search.py`](scikit-learn/model_evaluation/random_search.py)

## TensorFlow

### Basic Operations

- [`basic_operations/tensor_operations.py`](tensorflow/basic_operations/tensor_operations.py)
- [`basic_operations/variables_and_constants.py`](tensorflow/basic_operations/variables_and_constants.py)

### Neural Networks

- [`neural_networks/feedforward_nn.py`](tensorflow/neural_networks/feedforward_nn.py)
- [`neural_networks/convolutional_nn.py`](tensorflow/neural_networks/convolutional_nn.py)
- [`neural_networks/recurrent_nn.py`](tensorflow/neural_networks/recurrent_nn.py)

### Model Training

- [`model_training/compile_and_train.py`](tensorflow/model_training/compile_and_train.py)
- [`model_training/callbacks.py`](tensorflow/model_training/callbacks.py)

### Model Evaluation

- [`model_evaluation/evaluate_model.py`](tensorflow/model_evaluation/evaluate_model.py)
- [`model_evaluation/confusion_matrix.py`](tensorflow/model_evaluation/confusion_matrix.py)

### Model Saving and Loading

- [`model_saving_loading/save_and_load_model.py`](tensorflow/model_saving_loading/save_and_load_model.py)
- [`model_saving_loading/checkpoints.py`](tensorflow/model_saving_loading/checkpoints.py)

## PyTorch

### Basic Operations

- [`basic_operations/tensor_operations.py`](pytorch/basic_operations/tensor_operations.py)
- [`basic_operations/autograd.py`](pytorch/basic_operations/autograd.py)

### Neural Networks

- [`neural_networks/feedforward_nn.py`](pytorch/neural_networks/feedforward_nn.py)
- [`neural_networks/convolutional_nn.py`](pytorch/neural_networks/convolutional_nn.py)
- [`neural_networks/recurrent_nn.py`](pytorch/neural_networks/recurrent_nn.py)

### Model Training

- [`model_training/training_loop.py`](pytorch/model_training/training_loop.py)
- [`model_training/optimizers.py`](pytorch/model_training/optimizers.py)

### Model Evaluation

- [`model_evaluation/evaluate_model.py`](pytorch/model_evaluation/evaluate_model.py)
- [`model_evaluation/confusion_matrix.py`](pytorch/model_evaluation/confusion_matrix.py)

### Model Saving and Loading

- [`model_saving_loading/save_and_load_model.py`](pytorch/model_saving_loading/save_and_load_model.py)
- [`model_saving_loading/checkpoints.py`](pytorch/model_saving_loading/checkpoints.py)

## matplotlib

### Basic Plots

- [`basic_plots/line_plot.py`](matplotlib/basic_plots/line_plot.py)
- [`basic_plots/bar_chart.py`](matplotlib/basic_plots/bar_chart.py)
- [`basic_plots/histogram.py`](matplotlib/basic_plots/histogram.py)

### Advanced Plots

- [`advanced_plots/subplots.py`](matplotlib/advanced_plots/subplots.py)
- [`advanced_plots/3d_plot.py`](matplotlib/advanced_plots/3d_plot.py)
- [`advanced_plots/heatmap.py`](matplotlib/advanced_plots/heatmap.py)

### Customization

- [`customization/plot_styles.py`](matplotlib/customization/plot_styles.py)
- [`customization/annotations.py`](matplotlib/customization/annotations.py)
- [`customization/legends.py`](matplotlib/customization/legends.py)

## NumPy

### Array Operations

- [`array_operations/array_creation.py`](numpy/array_operations/array_creation.py)
- [`array_operations/array_manipulation.py`](numpy/array_operations/array_manipulation.py)

### Mathematical Operations

- [`math_operations/basic_math.py`](numpy/math_operations/basic_math.py)
- [`math_operations/statistics.py`](numpy/math_operations/statistics.py)

### Linear Algebra

- [`linear_algebra/matrix_multiplication.py`](numpy/linear_algebra/matrix_multiplication.py)
- [`linear_algebra/eigenvalues.py`](numpy/linear_algebra/eigenvalues.py)

### Random Sampling

- [`random_sampling/random_numbers.py`](numpy/random_sampling/random_numbers.py)
- [`random_sampling/distributions.py`](numpy/random_sampling/distributions.py)

## pandas

### DataFrame Operations

- [`dataframe_operations/creation.py`](pandas/dataframe_operations/creation.py)
- [`dataframe_operations/selection_and_filtering.py`](pandas/dataframe_operations/selection_and_filtering.py)

### Data Cleaning

- [`data_cleaning/handling_missing_data.py`](pandas/data_cleaning/handling_missing_data.py)
- [`data_cleaning/data_transformation.py`](pandas/data_cleaning/data_transformation.py)

### Data Aggregation

- [`data_aggregation/group_by.py`](pandas/data_aggregation/group_by.py)
- [`data_aggregation/pivot_table.py`](pandas/data_aggregation/pivot_table.py)

### Time Series Analysis

- [`time_series/creating_time_series.py`](pandas/time_series/creating_time_series.py)
- [`time_series/rolling_statistics.py`](pandas/time_series/rolling_statistics.py)
