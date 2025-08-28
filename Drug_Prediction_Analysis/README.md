# Drug Prediction Analysis

This project predicts the type of drug to be prescribed to patients based on medical attributes using various Machine Learning algorithms.

## Dataset
- 200 samples, 6 features (Age, Sex, BP, Cholesterol, Na_to_K ratio, Drug).
- Target variable: Drug (DrugA, DrugB, DrugC, DrugX, DrugY).

## Project Structure
```
Drug_Prediction_Analysis/
│── data/                # Dataset (CSV file)
│── notebooks/           # Jupyter notebooks
│── scripts/             # Python scripts for each model
│── results/             # Model outputs, evaluation metrics
│── images/              # Plots and visualizations
│── README.md            # Project documentation
```

## Implemented Models
- Logistic Regression
- Decision Tree
- KNN
- SVM
- K-Means (Clustering)
- Random Forest

## Performance Summary
- **Random Forest** achieved the best accuracy (100%).
- **Decision Tree & SVM** followed closely (97.5%).
- **Logistic Regression**: 90%.
- **KNN**: 85%.
- **K-Means**: 55% (since unsupervised).

## Example Visualizations
- Confusion Matrices
- Probability Distributions
- Model Comparison Charts
- Feature Importance (Random Forest)
- Clustering Visualization (K-Means)

## How to Run
1. Clone this repo:
   ```bash
   git clone <your_repo_link>
   cd Drug_Prediction_Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run models:
   ```bash
   python scripts/logistic_regression.py
   python scripts/decision_tree.py
   python scripts/knn.py
   python scripts/svm.py
   python scripts/kmeans.py
   python scripts/random_forest.py
   ```

4. View results inside `results/` and visualizations in `images/`.

