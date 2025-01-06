# Intro-ML-Basics: Introduction to Machine Learning with Scikit-Learn

This repository provides an introductory guide to machine learning using the **Scikit-Learn** library. It is designed for beginners and covers the full pipeline of a machine learning workflow, from **data preprocessing** to **model evaluation and visualization**. The project focuses on **classification tasks** using simple, well-known datasets and predefined models provided by Scikit-Learn.

---

## Features

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

---

## Project Structure

The repository consists of four Jupyter notebooks, each covering a specific stage of the machine learning workflow:

### 1. **Data Preprocessing**
   - Load datasets (`iris`, `wine`, `breast_cancer`) based on user input.
   - Handle missing values by either removing rows or imputing missing data.
   - Standardize the dataset to ensure features have a mean of 0 and standard deviation of 1.
   - Save preprocessed data for use in subsequent stages.

### 2. **Feature Engineering**
   - Apply **PCA** to reduce dimensionality.
   - Visualize the first two principal components with a scatter plot.
   - Save PCA-transformed data for model training.

### 3. **Model Selection**
   - Train multiple models using cross-validation.
   - Evaluate and compare models based on accuracy and standard deviation.
   - Save the best-performing model for final evaluation.

### 4. **Evaluation and Visualization**
   - Train the best model on the full training set.
   - Test the model on a hold-out test set.
   - Generate detailed evaluation metrics:
     - Classification report
     - Confusion matrix
     - ROC and Precision-Recall curves (if applicable)
     - Feature importance (if applicable).

---

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ml-intro-sklearn.git
   cd ml-intro-sklearn

2. **Run the Notebooks**:
   - Open the notebooks in your preferred Jupyter environment.
   - Follow the workflow step-by-step, starting from `Data Preprocessing`.

3. **Select a Dataset**:
   - Modify the `selected_dataset` variable in `Data Preprocessing` to choose between `iris`, `wine`, or `breast_cancer`.

4. **Customize Models**:
   - Modify the `selected_models` variable in `Model Selection` to train specific models.

5. **View Results**:
   - Check evaluation metrics and visualizations in the `Evaluation and Visualization` notebook.

---

## Considerations

1. **Avoiding Data Leakage**:
   - Data leakage occurs when information from the test set influences the training process, leading to overly optimistic evaluation metrics.
   - This repository avoids data leakage by:
     - Splitting data into training and test sets before evaluation.
     - Applying **standardization** and **PCA** only to the training set during cross-validation and then applying the same transformation to the test set.
     - Retraining the best model on the full training set before evaluating on the hold-out test set.

2. **Cross-Validation**:
   - The use of 5-fold stratified cross-validation ensures fair evaluation by splitting the dataset into folds that maintain class distribution.

3. **Dataset Size**:
   - The datasets used are small and primarily for educational purposes. They may not represent real-world complexities.

4. **Binary vs. Multi-Class Classification**:
   - Visualizations such as ROC and Precision-Recall curves are shown only for binary classification tasks. For multi-class datasets like `iris` and `wine`, these plots are skipped with a descriptive message.

5. **Feature Importance**:
   - Feature importance plots are generated only for models that support it (e.g., Random Forest and Gradient Boosting).

6. **Scalability**:
   - While this project demonstrates basic concepts, the code can be extended to handle larger datasets and more complex workflows with minimal adjustments.

---

## Dependencies

- Python 3.8 or higher
- Required libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tqdm`

Install dependencies using:
```bash
pip install -r requirements.txt
