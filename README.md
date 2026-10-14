# 📈 SVM Classification and Cross-Conformal Prediction

Tuned RBF Support Vector Machine classifiers on the Wine and USPS handwritten digits datasets, with a 10-fold cross-conformal predictor built on top of the SVM.

## 📌 Overview

The project has two parts:

1. **Classification:** an RBF SVM tuned with grid search and 5-fold cross-validation, comparing four feature-scaling methods on two very different datasets.
2. **Conformal prediction:** a cross-conformal predictor that gives a p-value for every class, validated with calibration curves and the average false p-value.

## ⚙️ How It Works

* **Baseline:** a default SVM, evaluated with 5-fold cross-validation on the training set and on a stratified 75/25 held-out test set.
* **Tuning:** `GridSearchCV` over `C` (0.1, 1, 10, 100) and `gamma` (0.001, 0.01, 0.1, 1), using a scikit-learn `Pipeline` so scaling is fitted inside each fold.
* **Scalers compared:** `StandardScaler`, `MinMaxScaler`, `RobustScaler` and `Normalizer`.
* **Cross-conformal prediction:** 10 folds. A model is trained on each fold's training part, and the held-out part is used for calibration. The nonconformity score is the highest SVM decision score among the other classes minus the score of the candidate class. P-values are pooled across folds.

## 📊 Results

| Dataset | Default SVM test error | Tuned SVM test error | Avg. false p-value |
|---------|------------------------|----------------------|--------------------|
| Wine | 33.3% | 0% (with feature scaling) | ≈ 0.008 |
| USPS | — | 2.3% | ≈ 0.004 |

**Key findings**
* **Wine:** `StandardScaler`, `MinMaxScaler` and `RobustScaler` all reached 0% test error, while `Normalizer` was much worse (about 36%). Wine's features sit on very different scales, and normalising each sample to unit length doesn't suit tabular data.
* **USPS:** `Normalizer` gave the lowest error, and the other three scalers landed between about 3.3% and 3.9%. Per-sample normalisation suits pixel data.
* **Calibration:** on both datasets the empirical error curves follow the ideal `y = x` line closely, so the cross-conformal predictor is approximately valid.

## 📂 Data

* **Wine:** loaded through scikit-learn
* **USPS handwritten digits:** included as `zip.train.gz` and `zip.test.gz` <!-- add the source of the data here -->

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** NumPy, scikit-learn, Matplotlib

## 🚀 Running the Project

1. Clone the repo:
```
   git clone https://github.com/andozaksu/svm-cross-conformal-prediction.git
   cd svm-cross-conformal-prediction
```
2. Install the libraries:
```
   pip install numpy scikit-learn matplotlib notebook
```
3. Open the notebook:
```
   jupyter notebook svm_cross_conformal_prediction.ipynb
```

The two `zip.*.gz` files must stay in the same folder as the notebook. The grid search runs several scalers on USPS, so it may take a few minutes.

## 👤 Author

**And Ozaksu**
[LinkedIn](https://www.linkedin.com/in/and-ozaksu-8241a7296) · [Email](mailto:andozaksu@gmail.com)