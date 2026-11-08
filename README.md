# Machine Learning Assignment 2 – Breast Cancer Classification

## a. Problem statement
Build and compare five supervised classification algorithms to predict whether a breast tumour is malignant or benign using diagnostic measurements. The models are trained once and saved. The Streamlit application evaluates a labelled test CSV using the saved models; it does not retrain on uploaded test data.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin Diagnostic (UCI Machine Learning Repository; distributed through scikit-learn).
- **Instances:** 569
- **Input features:** 30 numeric diagnostic features
- **Target:** `target` (0 = malignant, 1 = benign)
- **Task:** Binary classification
- **Split:** 80% training / 20% test, stratified, `random_state=42`.
- **Preprocessing:** StandardScaler is included in the Logistic Regression, KNN, and SVM pipelines. Decision Tree and Gaussian Naive Bayes use unscaled numeric features.
- **Test file:** `test_data.csv` contains the held-out test rows and target column.

## c. GitHub Repository Link
Add your repository URL here after creating and pushing this project to GitHub: **https://github.com/abdulaziz77798845/ml-assignement-2**

## d. Models used

The positive class for precision, recall, F1, and AUC is class 1 (benign). AUC is calculated from predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

- **Logistic Regression:** Accuracy = 0.9825, AUC = 0.9954, Precision = 0.9861, Recall = 0.9861, F1 = 0.9861, and MCC = 0.9623. This indicates strong performance on this held-out split.
- **Decision Tree:** Accuracy = 0.9211, AUC = 0.9163, Precision = 0.9565, Recall = 0.9167, F1 = 0.9362, and MCC = 0.8341. This is comparatively lower on this held-out split; inspect its precision/recall balance and confusion matrix.
- **KNN:** Accuracy = 0.9561, AUC = 0.9788, Precision = 0.9589, Recall = 0.9722, F1 = 0.9655, and MCC = 0.9054. This indicates strong performance on this held-out split.
- **Naive Bayes:** Accuracy = 0.9386, AUC = 0.9878, Precision = 0.9452, Recall = 0.9583, F1 = 0.9517, and MCC = 0.8676. This indicates strong performance on this held-out split.
- **Support Vector Machine:** Accuracy = 0.9825, AUC = 0.9950, Precision = 0.9861, Recall = 0.9861, F1 = 0.9861, and MCC = 0.9623. This indicates strong performance on this held-out split.

### Overall winner for this dataset

**Logistic Regression** is the winner on this experiment when comparing F1 first, then MCC and AUC as tie-breakers. This conclusion applies to the stated train/test split and random seed; different splits may produce different results.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository contents
- `app.py` – Streamlit evaluation interface
- `train_models.py` – reproducible training and export script
- `model/` – five saved fitted classifiers (including preprocessing pipelines where needed)
- `test_data.csv` – held-out labelled test data for the app
- `model_comparison.csv` and `metrics.json` – calculated evaluation results
- `requirements.txt` – application dependencies

## Streamlit Community Cloud deployment
1. Push the complete project to a GitHub repository.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Select **New app**, choose the repository and `main` branch.
4. Set the main file path to `app.py` and deploy.
5. Copy the live app URL into the final submission PDF.
6. https://ml-assignement-2-bqcwicykvappa3kfatws9lr.streamlit.app/ (Deployment URL)

## BITS Virtual Lab evidence
Run the training and evaluation workflow in BITS Virtual Lab and capture one screenshot showing the assignment execution. Add that screenshot to the final submission PDF. This screenshot and the live deployment must be completed by the student; they cannot be truthfully generated from this prepared project.
