# Machine Learning Course Notes (CSAI2020-2026)

This repository contains lecture notes, reference PDFs, and accompanying Python notebooks for the Machine Learning course.

Each topic includes:
- **Theory notes** (`.tex` and compiled `.pdf`) covering mathematical derivations, algorithms, and geometric intuition.
- **Accompanying notebooks** (`.ipynb`) with modular implementations, hyperparameter tuning guides, and code explanations.

---

## Contents (recommended reading order)

The folders keep their original names; the list below is the order to read them in.

### Mathematical foundations
* **Linear Algebra for ML** (`01_mathematical_foundations/01_linear_algebra/`)  
  Vectors, matrices, linear independence, projections, normal equations, spectral theorem, and Singular Value Decomposition (SVD).
* **Multivariate Calculus & Gradients** (`01_mathematical_foundations/02_multivariate_calculus/`)  
  Partial derivatives, gradients, directional derivatives, Hessians, Taylor series approximations, and Newton-Raphson vs. gradient descent.

### Appendices (NumPy and pandas: read before the data processing notes)
* **NumPy for Machine Learning** (`08_appendix/01_numpy_crash_course/`)  
  Creating, indexing, reshaping and aggregating arrays, broadcasting, vectorization, np.linalg, and why NumPy is fast.
* **Pandas Data Wrangling** (`08_appendix/02_pandas_data_wrangling/`)  
  Loading, inspecting, cleaning, grouping and merging tabular data; leak-free imputation; exploratory data analysis with plots; memory and dtypes.

* **Probability and Statistics** (`08_appendix/03_probability_and_statistics/`)  
  Random variables, PMF/CDF, expectation and variance, common distributions, Bayes' theorem, maximum likelihood, the central limit theorem, confidence intervals and hypothesis tests.

### Building a model
* **The Anatomy of an ML Problem** (`02_ml_fundamentals_and_pipelines/01_anatomy_of_a_model/`)  
  Model formulations, regression vs. classification losses, and vanishing gradient proofs under squared error for probabilities.
* **Data Processing Pipelines in Practice** (`02_ml_fundamentals_and_pipelines/03_data_processing_pipelines/`)  
  Missing value imputation, categorical encodings, scaling strategies, unit-circle cyclical date encoding, and leakage prevention.
* **Loss Functions & Evaluation Metrics** (`02_ml_fundamentals_and_pipelines/02_loss_functions_and_metrics/`)  
  Confusion matrix, Precision, Recall, $F_\beta$, ROC-AUC, PR curves, and classification thresholds.
* **Optimizers in Detail** (`02_ml_fundamentals_and_pipelines/04_optimizers_in_depth/`)  
  Batch, stochastic, and mini-batch gradient descent, Polyak's Momentum, RMSprop, and Adam with analytical bias correction.

### Linear models
* **Linear Regression from First Principles** (`03_linear_and_parametric_models/01_linear_regression/`)  
  Ordinary Least Squares derivation, geometric interpretations, Gauss-Markov theorem, multicollinearity, and statistical inference.
* **Regularization Mechanics (L1, L2, and ElasticNet)** (`03_linear_and_parametric_models/02_regularization_l1_l2/`)  
  Ridge ($L_2$) shrinkage, Lasso ($L_1$) sparsity, coordinate descent soft-thresholding, and bias-variance decomposition.
* **Logistic Regression & Binary Classification** (`03_linear_and_parametric_models/03_logistic_regression/`)  
  Log-odds, sigmoid link, maximum likelihood estimation, odds ratios, and probability cutoff optimization via KS statistic.

### Distance, probability and margin based models
* **K-Nearest Neighbors (KNN)** (`04_classical_and_instance_models/01_k_nearest_neighbors/`)  
  Distances, scaling and feature choice, the curse of dimensionality (with a simulation), choosing $K$, distance weighting, search trees, KNN regression, and why giving KNN every column hurts (bank data).
* **Naive Bayes Classification** (`04_classical_and_instance_models/02_naive_bayes/`)  
  Bayes' theorem, the independence assumption, Gaussian, multinomial and Bernoulli variants worked by hand, Laplace smoothing, overconfident probabilities, and classifying real consumer complaints.
* **Support Vector Machines (SVM)** (`04_classical_and_instance_models/03_support_vector_machines/`)  
  Maximum margin, soft margin and hinge loss (vs. logistic regression), the dual solved by hand, kernels by hand, what $C$ and $\gamma$ do, RBF SVM on the bank data, training time, and support vector regression.

### Trees and ensembles
* **Decision Trees from Scratch** (`05_tree_based_and_ensemble_models/01_decision_trees/`)  
  Recursive partitioning, Gini impurity, entropy, split criteria, and cost-complexity pruning ($ccp\_\alpha$).
* **Bagging, Random Forests & Extra Trees** (`05_tree_based_and_ensemble_models/02_bagging_rf_extra_trees/`)  
  Bootstrap aggregation, ensemble variance reduction ($\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$), Out-of-Bag (OOB) evaluation, feature subsampling, and MDI vs. permutation importance.
* **Cross-Validation & Hyperparameter Tuning** (`05_tree_based_and_ensemble_models/03_hyperparameter_tuning/`)  
  $K$-Fold, Stratified, Grouped, and Time-Series splits; Grid Search vs. Randomized Search and the Bergstra-Bengio theorem.
* **Gradient Boosting Machines (GBM)** (`06_boosting_machines/01_gradient_boosting_machines/`)  
  Fitting residuals tree by tree, gradient descent on the model, pseudo-residuals, regression and classification by hand, learning rate vs. number of trees, tree depth, subsampling and early stopping on validation data.
* **Extreme Gradient Boosting (XGBoost)** (`06_boosting_machines/02_xgboost/`)  
  The penalised objective, second-order Taylor expansion, best leaf values and split gain (worked by hand and checked against the library), parameters, regularisation experiments, scale_pos_weight, missing values, and importance.
* **CatBoost & Categorical Features** (`06_boosting_machines/03_catboost/`)  
  Target leakage in mean encoding (demonstrated on noise), ordered target statistics by hand, ordered boosting and prediction shift, symmetric trees, and CatBoost on the bank and loan data.

### Reading what a model learned
* **Model Inference and Interpretability** (`09_model_interpretability_and_inference/`)  
  Statistical inference in linear models (standard errors, $t$-statistics, $p$-values, confidence intervals, Cohen's $f^2$ effect sizes, odds ratios) and non-linear model explainability (Permutation Feature Importance, Partial Dependence Plots, ICE curves, LIME, and SHAP).

### Unsupervised learning
* **Distance Matrices and Hierarchical Clustering** (`07_unsupervised_learning/01_proximity_and_hierarchical/`)  
  Distances and scaling, agglomerative merging by hand, linkage criteria compared on different shapes, dendrograms, cophenetic correlation, and segmenting bank customers.
* **K-Means & PAM (K-Medoids)** (`07_unsupervised_learning/02_kmeans_and_pam/`)  
  Inertia, Lloyd's algorithm by hand, K-Means++, K-Medoids / PAM from scratch, elbow and silhouette, and why skewed features need transforming before segmenting customers.
* **DBSCAN & Density-Based Clustering** (`07_unsupervised_learning/03_dbscan/`)  
  Core, border and noise points by hand, the $k$-distance graph, the half-moons, clusters of different density (DBSCAN vs. HDBSCAN), and what DBSCAN says about the bank customers.
* **Principal Component Analysis (PCA)** (`07_unsupervised_learning/04_pca/`)  
  Maximum variance derivation, PCA on four points by hand, SVD, explained variance and scree plots, reconstruction, loadings on the bank data, and why PCA before a model can hurt.

### Neural networks
* **Feedforward Neural Networks** (`10_neural_networks/`)  
  Why non-linearity matters, XOR by hand, universal approximation, activation functions, backpropagation by hand (checked in NumPy), initialisation, regularisation, learning rate schedules and batch size, and MLPs on the bank and loan data with scikit-learn (no deep learning library needed).

---

## Datasets

All datasets referenced in notes and notebooks are located in the `data/` directory:
- `loan_data_train.csv` / `loan_data_test.csv`: Lending data used for regression and credit risk models.
- `bd_train.csv` / `bd_test.csv`: bank customer data (target `Revenue.Grid`) used for classification throughout, and for clustering and PCA. `post_code` and `post_area` are dropped everywhere (too many unique values).
- `insurance.csv`, `diabetic_data.csv`: additional datasets for exercises.
- `Consumer_Complaints_train.csv.zip`: consumer complaints with text, used in the Naive Bayes notes for text classification.

---

## Software

The notebooks use NumPy, pandas, matplotlib, scikit-learn, SciPy and statsmodels. The boosting and interpretability notebooks also need:

```bash
pip install xgboost catboost shap
```

## Building the PDFs

Each folder contains a standalone `.tex` file with its corresponding compiled `.pdf`. If you modify any LaTeX notes, you can compile all files using the automated script:

```bash
python3 compile_all.py
```

To recompile a single document:

```bash
cd 03_linear_and_parametric_models/01_linear_regression/
pdflatex -interaction=nonstopmode linear_regression.tex
pdflatex -interaction=nonstopmode linear_regression.tex
```
*(Running twice ensures tables of contents and cross-references resolve properly).*
