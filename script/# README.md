# README
# ----------------------------------------------------------------------------------------------------
# Project Title: Development of a Python Model to optimize Donor Funding Strategies at Anderson Cancer Center using Principal Component Analysis (PCA) and Logistic Regression techniques.
# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Created by: Victor Kadiri
# Date Created: 5th April 2025
# ----------------------------------------------
# Note: The Actual Analysis Starts from line 60 in the script
# ----------------------------------------------
# OBJECTIVE:
# The objective of the series of python script below is to perform a detailed analysis and visualisation to:
# a. To reduce the complexity of referral data by implementing Principal Component Analysis (PCA) through identification of the most critical variables influencing donor funding decisions. 
# b. To transform the dataset into two principal components in order to streamline the referral analysis process and enhance predictive modeling. 
# c. To implement logistic regression model to predict risk levels of cancer and prioritize high-risk patients for immediate attention. 
# ----------------------------------------------------------------------------------------------------
# KEY TERMS DEFINITIONS:
# - Principal Component Analysis(PCA): A technique that simplifies complex data by finding key patterns
# - Logistic Regression: A method to predict outcomes (like funding needs) from data
# - Malignant: Dangerous tumors requiring urgent treatment (red alerts)
# - Benign: Less dangerous tumors that may not need urgent treatment (yellow alerts)
# - Principal Components: The main patterns that explain differences in tumors
# - Decision Boundary: The line that separates risky and safe cases
# - ROC-AUC (Receiver Operating Characteristic - Area Under the Curve): How good the computer is at telling risky and safe cases apart
# - Precision-Recall: How often the computer's warnings are correct
# -----------------------------------------------------------------------------------------------------
# PROCEDURES:
# The procedures performed are as follows:
# - 1. INITIALIZATION & IMPORTS: Import required Python libraries
# - 2. FOLDER PATH SETUP: Define output directories: data, models, reports, and visualizations.
# - 3. DATA LOADING WITH ERROR HANDLING: Import the cancer dataset from sklearn.datasets with exception handling considerations.
# - 4. DATA CLEANING WITH VALIDATION: Handle missing values, outliers, and standardize features for PCA.
# - 5. PCA ANALYSIS WITH VISUALIZATION: Reduce dimensions to 2 principal components and plot tumor patterns.
# - 6. PREDICTIVE MODELING: Train a logistic regression model on PCA-transformed data
# - 7. REPORT GENERATION: Generate actionable insights for the center (clinical_report.txt) and donors (funding_proposal.txt).
# - 8. MODEL VALIDATION AND TESTING: The code performs detailed model validation and testing 
# - 9. KEY FINDINGS SUMMARY: Creates a summary report of key insights
# - 10. MAIN EXECUTION: Execute the full pipeline and export results.
# -----------------------------------------------------------------------------------------------------
# DESIRED OUTPUT: 
# The outputs of the python model will be:
# ├── data/
# │   └── patient_data_clean.csv: Cleaned and standardized dataset for reproducibility
# ├── models/
# │   └── risk_predictor.pkl: Saved model for future predictions.
# ├── reports/
# │   ├── clinical_report.txt: Summary report showing high-risk patients for clinicians.
# │   ├── donor_proposal.txt: Summary report data-driven case for funding allocation.
# │   ├── model_performance.txt: Summary of model performance evaluation
# │   ├── model_validation.txt: Summary of model validation and deployment guidance
# │   └── key_findings_summary.txt: Summary of key findings
# └── visualizations/
#     ├── tumor_patterns.png: Visualizes PCA components separating malignant/benign cases
#     └── confusion_matrix.png: Visualisation of model performance showing correct/incorrect predictions
#     ├── feature_importance.png: visualizes original features that contribute most to predictions through PCA components.
#     ├── decision_boundary.png: visualizes how the model separates high-risk and low-risk cases.
#     └── precision_recall_curve.png: visualizes the threshold analysis graph showing tradeoff between precision/recall and area under curve (AUC)
# ----------------------------------------------------------------------------------------------------

# Prerequisites
# i. Datasets Required
# Import the cancer dataset from sklearn.datasets
# ii. Scripts Required: 
# a. Cancer_Analysis_Funding_optimization.py
# iii. Tools and Installations Required 
# - For python analysis and visualization:
# 1. Download and Install VS Code 
# 2. Download and Install Python 3.12.9 or higher
# 3. ## Additional Libraries Required
# The following Python libraries are used in this analysis and need to be installed:
### Core Libraries:
# ----- `numpy` (v1.26.0+): For numerical operations and array handling
# ----- `pandas` (v2.1.0+): For data manipulation and analysis
# ----- `matplotlib` (v3.8.0+): For creating static visualizations
# ----- `seaborn` (v0.13.0+): For enhanced statistical visualizations
# ----- `scikit-learn` (v1.3.0+): For machine learning and statistical modeling
### Specialized Libraries:
# ----- `scikit-learn` components:
# ----- `sklearn.datasets.load_breast_cancer`: For the cancer dataset
# ----- `sklearn.preprocessing.StandardScaler`: For feature scaling
# ----- `sklearn.decomposition.PCA`: For principal component analysis
# ----- `sklearn.linear_model.LogisticRegression`: For predictive modeling
# ----- `sklearn.model_selection.train_test_split`: For data splitting
# ----- `sklearn.metrics`: For model evaluation metrics
### Installation Command:
# ----  ```bash
# ----  pip install numpy pandas matplotlib seaborn scikit-learn

# Step by step procedures Performed:
# 1: Launch the VS Code application
# 2: Create a main folder: "Cancer_Analysis" containing two sub folders( i. "output" and  ii. "scripts") in your desired location. 
# 3: Download all files from the GIT repos into the relevant sub folders above.
# 4: Locate and the main folder: "Cancer_Analysis" using VS code
# 5: Locate and open the script file (a.Cancer_Analysis_Funding_optimization.py) within VS code environment.
# 6: Once opened, locate and Run the script using Ctrl+A and then Shift+Enter or click Run within the VS code environment
# 7: Update the output path when prompted
# 8: View the outputs on the terminal panes and saved results in the output folder.

# Support:
#   For questions or issues, please open an issue on the GitHub repository.

# Note:
# Refer to the python scripts for the step by step understanding of the model.
# Refer to file "Vs_code_folder_structure.png" for the folder structure used in VS code for this analysis.
