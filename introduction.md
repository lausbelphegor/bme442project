
# 1. INTRODUCTION

## Problem Statement

Electroencephalography is a non-invasive method used to record electrical activity of the brain. EEG-based Motor Imagery is necessary in developing Brain-Computer Interfaces, where individuals imagine performing specific motor tasks. Accurately classifying these imagined tasks from EEG signals is challenging due to the complex and noisy nature of EEG data.

## Project Aim

This project aims to develop an efficient and accurate classification model for EEG-based Motor Imagery tasks using the EEGNet model, a convolutional neural network designed for EEG signal classification. The main objectives are:

1. **Data Preprocessing**: Filter and segment raw EEG data from the EEG Motor Movement/Imagery Dataset (EEGMMIDB).
2. **Model Development**: Train the EEGNet model to classify EEG patterns corresponding to different MI tasks.
3. **Evaluation**: Assess model performance using stratified k-fold cross-validation and metrics such as accuracy, sensitivity, specificity, PPV, and NPV.
4. **Optimization**: Optimize hyperparameters and implement learning rate scheduling to enhance performance.

This project aims to experiment with deep learning methods with EEG-MI data, potentially aiding individuals in controlling external devices through brain activity.

