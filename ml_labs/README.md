# Machine Learning Lab

Fourteen lab experiments, one folder each. Every folder holds a notebook with complete, working code and step-by-step explanations, and a `data/` folder with the dataset exactly as downloaded: the cleaning is part of the lab.

## How to work through a lab

1. Read the notes on the methods the lab uses first (the notebook mentions the ideas it relies on: regularisation, TF-IDF, K-means, cross-validation and so on).
2. Open the notebook from inside its own folder, so that the relative path `data/...` works, and run it cell by cell. Read the explanation above each cell before running it, and look at what each output says before moving on.
3. Then change things: other features, other hyperparameters, another cutoff. The "things to try" at the end of several labs are good starting points.

## The labs

| Lab | Topic | Methods | Dataset in this repo? |
|---|---|---|---|
| 1 | Used car price prediction | EDA, linear, Ridge, Lasso, polynomial regression, residuals | yes |
| 2 | Air quality (PM2.5) prediction | time-ordered split, lag features, linear/Ridge/Lasso, feature selection | yes |
| 3 | Credit card fraud detection | imbalance, class weights, undersampling, logistic regression, SVM, ROC, KS cutoff | **no, download (150 MB)** |
| 4 | Online news popularity | target definition, multicollinearity, skew, mutual information, logistic/tree/RF/GBM | yes |
| 5 | Product review sentiment | text preprocessing, TF-IDF, Naive Bayes, logistic regression, linear SVM, ablation | **no, download (690 MB)** |
| 6 | Fake job posting detection | structured + text features, logistic regression, Naive Bayes, random forest | yes |
| 7 | Customer segmentation | K-means, K-prototypes, elbow, silhouette, PCA, persona cards | yes |
| 8 | Human activity recognition | k-NN (Euclidean/Manhattan), linear SVM, MLP, confusion matrix | **no, download (300 MB)** |
| 9 | Household energy forecasting | resampling, lag and rolling features, linear/Ridge/Lasso/polynomial, forecast horizon | yes |
| 10 | Email spam and phishing detection | TF-IDF, header features, Naive Bayes, logistic regression, error analysis | yes |
| 11 | Movie recommendations | rating matrix, cosine similarity, user-user and item-item neighbours, RMSE, hit-rate | yes |
| 12 | News topic clustering | TF-IDF, cosine similarity, K-means, LSA, hierarchical clustering, silhouette | yes |
| 13 | IoT network anomaly detection | K-means distance scores, thresholds, logistic regression, random forest | **no, download (300 MB)** |
| 14 | Hate speech detection | tweet preprocessing, n-grams, multiclass logistic regression, Naive Bayes, macro-F1, ethics | yes |

**Datasets that are too big for git** (labs 3, 5, 8 and 13): the notebook starts with the download link and says where to put the files. Download them before the lab session; the Amazon reviews (lab 5) take a while.

Lab 14 contains offensive language from real tweets; the notebook masks slurs when displaying text.

## Software

NumPy, pandas, matplotlib, seaborn, scikit-learn and NLTK (for the Porter stemmer). Lab 7 also uses the `kmodes` package for K-prototypes:

```bash
pip install kmodes
```

If pandas warns that your `numexpr` is too old, update it (`pip install -U numexpr`): with some version combinations pandas can silently compute wrong results on large tables.
