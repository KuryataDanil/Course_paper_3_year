# Course Paper: Plant Seedlings Classification

## Overview

This repository contains the coursework of a student from Southern Federal University (SFedU) for the 3rd year, focusing on the topic **"Solving the problem of classifying plant seedlings by their images"**. The project leverages deep learning techniques to develop an automated system for recognizing plant seedlings based on image data. The solution is implemented using a convolutional neural network (ResNet-18) trained on the "Plant Seedlings Classification" dataset from Kaggle.

### Project Goals
- Develop a machine learning model to classify plant seedlings accurately.
- Perform data preprocessing (resizing, augmentation, normalization) to enhance model performance.
- Achieve high accuracy on the validation set and analyze classification errors.
- Provide a foundation for automated crop monitoring in agriculture.

## Dataset
The project utilizes the **Plant Seedlings Classification** dataset, available on Kaggle:
- **Source**: [https://www.kaggle.com/competitions/plant-seedlings-classification/data](https://www.kaggle.com/competitions/plant-seedlings-classification/data)
- **Description**: This dataset contains images of various plant seedlings, labeled with their species, suitable for training and evaluating image classification models.

## Features
- **Model**: ResNet-18 convolutional neural network.
- **Accuracy**: Achieved 92.21% on the validation set.
- **Preprocessing**: Includes image resizing, data augmentation, and normalization.
- **Analysis**: Identification of problematic classes and suggestions for model improvement.

## Installation

### Prerequisites
- Python 3.9 or higher
- Required libraries: `numpy`, `pandas`, `torch`, `torchvision`, `matplotlib`, `scikit-learn`

### Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Course_paper_3_year.git
   cd Course_paper_3_year