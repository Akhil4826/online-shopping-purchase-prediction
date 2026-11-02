# Online Shopping Purchase Prediction

BITS Pilani WILP M.Tech (AIML) — Machine Learning Assignment 2

## A. Problem Statement
This project predicts whether a visitor will complete a purchase during an online shopping session using browsing and session behavior data. Five classification models are trained and compared on held-out test data to see which model handles purchase prediction best.

## B. Dataset Description
- **Dataset Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository (ID: 468)
- **Total Records:** 12,330 sessions
- **Input Features:** 17 features (10 numerical features such as durations, bounce rates, exit rates, and page values; 7 categorical features such as month, visitor type, region, and operating system)
- **Target Variable:** Revenue (True = purchase made, False = no purchase)
- **Class Distribution:** 10,422 False (84.53%) and 1,908 True (15.47%)

## C. GitHub Repository Link
https://github.com/Akhil4826/online-shopping-purchase-prediction

**Live Streamlit Application:**
https://online-shopping-purchase-prediction.streamlit.app/

## D. Models Used
1. **Logistic Regression:** A linear model predicting purchase probability using regularized log-odds.
2. **Decision Tree Classifier:** A tree-based model splitting features recursively by Gini impurity.
3. **K-Nearest Neighbors (KNN):** An instance-based classifier using distances to the 5 nearest neighbors.
4. **Naive Bayes (Gaussian):** A probabilistic classifier applying Bayes' rule assuming feature independence.
5. **Support Vector Machine (LinearSVC):** A linear classifier finding the maximum margin decision boundary.

## E. Model Comparison
The table below shows the evaluation results on the 20% held-out test set (2,466 records). All metrics use `Revenue = True` as the positive class:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8585 | 0.7356 | 0.5420 | 0.5576 | 0.5497 | 0.4658 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Support Vector Machine | 0.8788 | 0.8960 | 0.7712 | 0.3089 | 0.4411 | 0.4381 |

## F. Observations
- **Logistic Regression:** Achieved high accuracy (88.08%) and good precision (73.66%), but had low recall (35.86%) because it predicts the positive minority class conservatively.
- **Decision Tree:** Balanced positive-class precision (54.20%) and recall (55.76%), leading to the highest F1 score (0.5497) and highest MCC (0.4658).
- **KNN:** Showed decent overall accuracy (87.67%) and precision (68.22%), but achieved modest recall (38.22%) on the minority class.
- **Naive Bayes:** Detected nearly all buyers (97.38% recall), but produced too many false positives because browsing features are correlated, dropping accuracy to 27.29%.
- **Support Vector Machine:** Achieved the highest AUC (0.8960) and highest precision (77.12%), making it the most reliable model when false positives must be avoided.

## G. Overall Winner
**Winner: Decision Tree Classifier**

Because this dataset is imbalanced (only ~15.5% purchases), overall accuracy is dominated by the negative class. The Decision Tree is the overall winner because it achieves the highest positive-class F1 score (0.5497) and highest MCC (0.4658), identifying over 55% of actual shoppers who buy while maintaining reasonable precision.

---

### Running the Project Locally
```bash
git clone https://github.com/Akhil4826/online-shopping-purchase-prediction.git
cd online-shopping-purchase-prediction
pip install -r requirements.txt
streamlit run app.py
```
