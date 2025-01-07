# Intro-ML-Basics: Introduction to Machine Learning with Scikit-Learn

This repository provides an introductory guide to machine learning using the **Scikit-Learn** library. It is designed for beginners and covers a basic pipeline of a simple machine learning workflow, from **data preprocessing** to **model evaluation and visualization**. The project includes workflows for both **classification** and **regression tasks**, using simple, well-known datasets and predefined models provided by Scikit-Learn. As this is an introduction to machine learning, more advanced topics including statistical evaluation of models, hyperparameter-tuning, feature selection, ensemple methods and analysis of more advanced datasets will be covered in a later project.

---

## Features

### Classification
1. **Dataset Support**:
   - Iris: Flower classification with 3 classes and 4 features.
   - Wine: Wine cultivar classification with 3 classes and 13 features.
   - Breast Cancer: Tumor classification with 2 classes and 30 features.

2. **Complete Workflow**:
   - **Data Preprocessing**: Load datasets, handle missing values, standardize features.
   - **Feature Engineering**: Apply **Principal Component Analysis (PCA)** to reduce dimensionality.
   - **Model Selection**: Train and evaluate predefined models with cross-validation.
   - **Evaluation & Visualization**: Assess the best model on a hold-out test set with detailed visualizations.

3. **Predefined Models**:
   - Logistic Regression
   - Support Vector Machines (SVM)
   - Random Forest
   - Gradient Boosting
   - Simple Neural Network (MLPClassifier)

4. **Visualization Tools**:
   - PCA Scatter Plot
   - Confusion Matrix
   - Cross-Validation Results
   - ROC and Precision-Recall Curves (for binary classification)
   - Feature Importance (if applicable)

### Regression
1. **Dataset Support**:
   - Diabetes: Predict disease progression based on 10 features.
   - California Housing: Predict house prices based on 8 features.

2. **Complete Workflow**:
   - **Data Preprocessing**: Load datasets, handle missing values, standardize features.
   - **Feature Engineering**: Apply **Principal Component Analysis (PCA)** to reduce dimensionality.
   - **Model Selection**: Train and evaluate predefined models with cross-validation.
   - **Evaluation & Visualization**: Assess the best model on a hold-out test set with detailed visualizations.

3. **Predefined Models**:
   - Linear Regression
   - Ridge and Lasso Regression
   - Support Vector Regression (SVR)
   - Random Forest Regressor
   - Gradient Boosting Regressor

4. **Visualization Tools**:
   - PCA Scatter Plot
   - True vs. Predicted Values
   - Residual Plot
   - Feature Importance (if applicable)

---

## Project Structure

The repository is divided into two main folders:

1. **Classification**:
   - Contains notebooks for classification tasks, focusing on datasets like Iris, Wine, and Breast Cancer.
   - Includes the following notebooks:
     - `1_Data_Preprocessing.ipynb`
     - `2_Feature_Engineering.ipynb`
     - `3_Model_Selection.ipynb`
     - `4_Evaluation_and_Visualization.ipynb`

2. **Regression**:
   - Contains notebooks for regression tasks, focusing on datasets like Diabetes and California Housing.
   - Includes the following notebooks:
     - `1_Data_Preprocessing.ipynb`
     - `2_Feature_Engineering.ipynb`
     - `3_Model_Selection.ipynb`
     - `4_Evaluation_and_Visualization.ipynb`

---

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NicholasBorch/Intro-ML-Basics.git
   cd Intro-ML-Basics

2. **Choose a Folder**:
   - Navigate to the `Classification` or `Regression` folder based on your interest.

3. **Run the Notebooks**:
   - Open the notebooks in your preferred Jupyter environment.
   - Follow the workflow step-by-step, starting from `Data_Preprocessing`.

4. **Select a Dataset**:
   - Modify the `selected_dataset` variable in `Data_Preprocessing` to choose an appropriate dataset.

5. **Customize Models**:
   - Modify the `selected_models` variable in `Model_Selection` to train specific models.

6. **View Results**:
   - Check evaluation metrics and visualizations in the `Evaluation_and_Visualization` notebook.

---

## Considerations

1. **Avoiding Data Leakage**:
   - This repository avoids data leakage by:
     - Splitting data into training and test sets before evaluation.
     - Applying **standardization** and **PCA** only to the training set during cross-validation and then applying the same transformation to the test set.
     - Retraining the best model on the full training set before evaluating on the hold-out test set.

2. **Cross-Validation**:
   - Uses 5-fold cross-validation for fair evaluation of models.

3. **Dataset Size**:
   - The datasets used are small and primarily for educational purposes. They may not represent real-world complexities.

4. **Scalability**:
   - While this project demonstrates basic concepts, the code can be extended to handle larger datasets and more complex workflows with minimal adjustments.

---

## Dependencies

Install dependencies using:
```bash
pip install -r requirements.txt
