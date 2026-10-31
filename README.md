# Machine Learning Course Notes (CSAI2020-2026)

This repository contains lecture notes, reference PDFs, and accompanying Python notebooks for the Machine Learning course.

Each module includes:
- **Theory notes** (`.tex` and compiled `.pdf`) covering mathematical derivations, algorithms, and geometric intuition.
- **Accompanying notebooks** (`.ipynb`) with modular implementations and code explanations.

---

## Course Index & Structure

### Part 1: Mathematical Foundations
* **Unit 1.1 (Chapter 1): Linear Algebra for ML** (`01_mathematical_foundations/01_linear_algebra/`)  
  Vectors, matrices, linear independence, projections, normal equations, and eigendecomposition.
* **Unit 1.2 (Chapter 2): Multivariate Calculus & Gradients** (`01_mathematical_foundations/02_multivariate_calculus/`)  
  Partial derivatives, gradients, Hessians, Taylor series approximations, and gradient descent.

### Part 2: ML Fundamentals & Data Processing
* **Unit 2.1 (Chapter 3): The Anatomy of an ML Problem** (`02_ml_fundamentals_and_pipelines/01_anatomy_of_a_model/`)  
  Model formulations, regression vs. classification losses, and why squared error fails for probabilities.
* **Unit 2.2 (Chapter 4): Loss Functions & Evaluation Metrics** (`02_ml_fundamentals_and_pipelines/02_loss_functions_and_metrics/`)  
  Confusion matrix, Precision, Recall, $F_\beta$, ROC-AUC, PR curves, and classification thresholds.
* **Unit 2.3 (Chapter 5): Data Processing Pipelines** (`02_ml_fundamentals_and_pipelines/03_data_processing_pipelines/`)  
  Missing value imputation, categorical encodings, scaling strategies, and cyclical date encoding.
* **Unit 2.4 (Chapter 6): Optimizers in Detail** (`02_ml_fundamentals_and_pipelines/04_optimizers_in_depth/`)  
  Batch, stochastic, and mini-batch gradient descent, Momentum, and Adam.

### Part 3: Linear & Parametric Models
* **Unit 3.1 (Chapter 7): Linear Regression from First Principles** (`03_linear_and_parametric_models/01_linear_regression/`)  
  Ordinary Least Squares derivation, geometric interpretations, and multicollinearity diagnostics.
* **Unit 3.2 (Chapter 8): Regularization & Bias-Variance** (`03_linear_and_parametric_models/02_regularization_l1_l2/`)  
  Ridge ($L_2$) shrinkage, Lasso ($L_1$) sparsity, and the geometry of regularization penalties.
* **Unit 3.3 (Chapter 9): Logistic Regression & Binary Classification** (`03_linear_and_parametric_models/03_logistic_regression/`)  
  Log-odds, sigmoid link, maximum likelihood estimation, and cutoff selection via KS / Youden's $J$.

### Part 4: Classical & Instance-Based Models
* **Unit 4.1 (Chapter 10): K-Nearest Neighbors (KNN)** (`04_classical_and_instance_models/01_k_nearest_neighbors/`)  
  Distance metrics, neighbor weighting, decision boundaries, and scaling sensitivity.
* **Unit 4.2 (Chapter 11): Naive Bayes Classification** (`04_classical_and_instance_models/02_naive_bayes/`)  
  Conditional independence, Laplace smoothing, and Multinomial vs. Gaussian models.
* **Unit 4.3 (Chapter 12): Support Vector Machines (SVM)** (`04_classical_and_instance_models/03_support_vector_machines/`)  
  Maximum margin hyperplanes, soft-margin slack ($C$), support vector sparsity, and kernel functions.

### Part 5: Tree-Based Models & Bagging Ensembles
* **Unit 5.1 (Chapter 13): Decision Trees from Scratch** (`05_tree_based_and_ensemble_models/01_decision_trees/`)  
  Recursive partitioning, Gini impurity, entropy, split criteria, and tree constraints.
* **Unit 5.2 (Chapter 14): Bagging, Random Forests & Extra Trees** (`05_tree_based_and_ensemble_models/02_bagging_rf_extra_trees/`)  
  Bootstrap aggregation, variance reduction through averaging, Out-of-Bag (OOB) evaluation, feature subsampling, and permutation feature importance.
* **Unit 5.3 (Chapter 15): Cross-Validation & Hyperparameter Tuning** (`05_tree_based_and_ensemble_models/03_hyperparameter_tuning/`)  
  $K$-Fold, Stratified, Grouped, and Time-Series splits; Grid Search vs. Randomized Search.

### Part 6: Boosting Machines
* **Unit 6.1 (Chapter 16): Gradient Boosting Machines (GBM)** (`06_boosting_machines/01_gradient_boosting_machines/`)  
  Functional gradient descent, pseudo-residuals, weak learners, shrinkage, and stochastic boosting.
* **Unit 6.2 (Chapter 17): Extreme Gradient Boosting (XGBoost)** (`06_boosting_machines/02_xgboost/`)  
  Second-order Taylor loss expansion, regularized tree objectives, optimal leaf weights, and split gain.
* **Unit 6.3 (Chapter 18): CatBoost & Categorical Features** (`06_boosting_machines/03_catboost/`)  
  Target leakage in mean encoding, Ordered Target Statistics, and symmetric oblivious trees.

### Part 7: Unsupervised Learning
* **Unit 7.1 (Chapter 19): Hierarchical Clustering** (`07_unsupervised_learning/01_proximity_and_hierarchical/`)  
  Pairwise proximity matrices, agglomerative merging, linkage criteria (Single, Complete, Average, Ward), and dendrogram pruning.
* **Unit 7.2 (Chapter 20): K-Means & PAM (K-Medoids)** (`07_unsupervised_learning/02_kmeans_and_pam/`)  
  Within-cluster sum of squares (inertia), Lloyd's algorithm, K-Means++, K-Medoids / PAM for outlier robustness, the Elbow method, and Silhouette analysis.
* **Unit 7.3 (Chapter 21): DBSCAN & Density-Based Clustering** (`07_unsupervised_learning/03_dbscan/`)  
  $\epsilon$-neighborhoods, MinPts, core/border/noise points, density-connectivity, and the $k$-distance graph.
* **Unit 7.4 (Chapter 22): Principal Component Analysis (PCA)** (`07_unsupervised_learning/04_pca/`)  
  Variance maximization, covariance eigendecomposition, orthogonal projections, and scree plots.

### Part 8: Appendices
* **Appendix A: NumPy for Machine Learning** (`08_appendix/01_numpy_crash_course/`)  
  Array memory layout, striding, vectorized arithmetic, and broadcasting rules.
* **Appendix B: Pandas Data Wrangling** (`08_appendix/02_pandas_data_wrangling/`)  
  Series and DataFrames, BlockManager internals, missing data handling, and leakage-free transformations.

---

## Datasets

All datasets referenced in notes and notebooks are located in the `data/` directory:
- `loan_data_train.csv` / `loan_data_test.csv`: Lending data used for regression and credit risk models.
- `bd_train.csv` / `bd_test.csv`: Bank marketing campaign data used for classification, KS metrics, and cutoffs.
- `insurance.csv`: Health charges used for regression analysis and feature transformations.
- `diabetic_data.csv`: Clinical records used for categorical encoding and tree models.
- `Consumer_Complaints_train.csv.zip`: Text complaints data used for NLP and text classification.

---

## Building the PDFs

Each folder contains a standalone `.tex` file with its corresponding compiled `.pdf`. If you modify any LaTeX notes, you can recompile individual files:

```bash
cd 03_linear_and_parametric_models/01_linear_regression/
pdflatex -interaction=nonstopmode linear_regression.tex
pdflatex -interaction=nonstopmode linear_regression.tex
```
*(Running twice ensures tables of contents and cross-references resolve properly).*
