# Machine Learning Course Notes (CSAI2020-2026)

This repository contains lecture notes, reference PDFs, and accompanying Python notebooks for the Machine Learning course.

Each module includes:
- **Theory notes** (`.tex` and compiled `.pdf`) covering mathematical derivations, algorithms, and geometric intuition.
- **Accompanying notebooks** (`.ipynb`) with modular implementations, hyperparameter tuning guides, and code explanations.

---

## Course Index & Structure

### Part 1: Mathematical Foundations
* **Unit 1.1: Linear Algebra for ML** (`01_mathematical_foundations/01_linear_algebra/`)  
  Vectors, matrices, linear independence, projections, normal equations, spectral theorem, and Singular Value Decomposition (SVD).
* **Unit 1.2: Multivariate Calculus & Gradients** (`01_mathematical_foundations/02_multivariate_calculus/`)  
  Partial derivatives, gradients, directional derivatives, Hessians, Taylor series approximations, and Newton-Raphson vs. gradient descent.

### Part 2: ML Fundamentals & Data Processing
* **Unit 2.1: The Anatomy of an ML Problem** (`02_ml_fundamentals_and_pipelines/01_anatomy_of_a_model/`)  
  Model formulations, regression vs. classification losses, and vanishing gradient proofs under squared error for probabilities.
* **Unit 2.2: Loss Functions & Evaluation Metrics** (`02_ml_fundamentals_and_pipelines/02_loss_functions_and_metrics/`)  
  Confusion matrix, Precision, Recall, $F_\beta$, ROC-AUC, PR curves, and classification thresholds.
* **Unit 2.3: Data Processing Pipelines in Practice** (`02_ml_fundamentals_and_pipelines/03_data_processing_pipelines/`)  
  Missing value imputation, categorical encodings, scaling strategies, unit-circle cyclical date encoding, and leakage prevention.
* **Unit 2.4: Optimizers in Detail** (`02_ml_fundamentals_and_pipelines/04_optimizers_in_depth/`)  
  Batch, stochastic, and mini-batch gradient descent, Polyak's Momentum, RMSprop, and Adam with analytical bias correction.

### Part 3: Linear & Parametric Models
* **Unit 3.1: Linear Regression from First Principles** (`03_linear_and_parametric_models/01_linear_regression/`)  
  Ordinary Least Squares derivation, geometric interpretations, Gauss-Markov theorem, multicollinearity, and statistical inference.
* **Unit 3.2: Regularization Mechanics (L1, L2, and ElasticNet)** (`03_linear_and_parametric_models/02_regularization_l1_l2/`)  
  Ridge ($L_2$) shrinkage, Lasso ($L_1$) sparsity, coordinate descent soft-thresholding, and bias-variance decomposition.
* **Unit 3.3: Logistic Regression & Binary Classification** (`03_linear_and_parametric_models/03_logistic_regression/`)  
  Log-odds, sigmoid link, maximum likelihood estimation, odds ratios, and probability cutoff optimization via KS statistic.

### Part 4: Classical & Instance-Based Models
* **Unit 4.1: K-Nearest Neighbors (KNN)** (`04_classical_and_instance_models/01_k_nearest_neighbors/`)  
  Distance metrics, neighbor weighting, decision boundaries, curse of dimensionality, and hyperparameter tuning.
* **Unit 4.2: Naive Bayes Classification** (`04_classical_and_instance_models/02_naive_bayes/`)  
  Conditional independence, Laplace smoothing, Dirichlet priors, and Multinomial vs. Gaussian models.
* **Unit 4.3: Support Vector Machines (SVM)** (`04_classical_and_instance_models/03_support_vector_machines/`)  
  Maximum margin hyperplanes, soft-margin slack ($C$), dual Lagrangian formulation, and Mercer's kernel trick (RBF).

### Part 5: Tree-Based Models & Bagging Ensembles
* **Unit 5.1: Decision Trees from Scratch** (`05_tree_based_and_ensemble_models/01_decision_trees/`)  
  Recursive partitioning, Gini impurity, entropy, split criteria, and cost-complexity pruning ($ccp\_\alpha$).
* **Unit 5.2: Bagging, Random Forests & Extra Trees** (`05_tree_based_and_ensemble_models/02_bagging_rf_extra_trees/`)  
  Bootstrap aggregation, ensemble variance reduction ($\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$), Out-of-Bag (OOB) evaluation, feature subsampling, and MDI vs. permutation importance.
* **Unit 5.3: Cross-Validation & Hyperparameter Tuning** (`05_tree_based_and_ensemble_models/03_hyperparameter_tuning/`)  
  $K$-Fold, Stratified, Grouped, and Time-Series splits; Grid Search vs. Randomized Search and the Bergstra-Bengio theorem.

### Part 6: Boosting Machines
* **Unit 6.1: Gradient Boosting Machines (GBM)** (`06_boosting_machines/01_gradient_boosting_machines/`)  
  Functional gradient descent in $L_2$ Hilbert space, pseudo-residuals, shrinkage, and Newton-Raphson leaf updates.
* **Unit 6.2: Extreme Gradient Boosting (XGBoost)** (`06_boosting_machines/02_xgboost/`)  
  Second-order Taylor loss expansion, regularized tree objectives, optimal leaf weights, split gain, and cache-aware execution.
* **Unit 6.3: CatBoost & Categorical Features** (`06_boosting_machines/03_catboost/`)  
  Target leakage in mean encoding, Ordered Target Statistics, and symmetric oblivious trees with bitwise SIMD evaluation.

### Part 7: Unsupervised Learning
* **Unit 7.1: Distance Matrices and Hierarchical Clustering** (`07_unsupervised_learning/01_proximity_and_hierarchical/`)  
  Pairwise proximity matrices, agglomerative merging, linkage criteria (Single, Complete, Average, Ward), Lance-Williams recurrence, and dendrogram pruning.
* **Unit 7.2: K-Means & PAM (K-Medoids)** (`07_unsupervised_learning/02_kmeans_and_pam/`)  
  Within-cluster sum of squares (inertia), Lloyd's algorithm, K-Means++, K-Medoids / PAM for outlier robustness, the Elbow method, and Silhouette analysis.
* **Unit 7.3: DBSCAN & Density-Based Clustering** (`07_unsupervised_learning/03_dbscan/`)  
  $\epsilon$-neighborhoods, MinPts, core/border/noise points, density-connectivity, and the $k$-distance graph.
* **Unit 7.4: Principal Component Analysis (PCA)** (`07_unsupervised_learning/04_pca/`)  
  Variance maximization, covariance eigendecomposition, orthogonal projections, Singular Value Decomposition (SVD), and scree plots.

### Part 8: Appendices
* **Appendix A: NumPy for Machine Learning** (`08_appendix/01_numpy_crash_course/`)  
  Array memory layout, striding, vectorized arithmetic, and broadcasting rules.
* **Appendix B: Pandas Data Wrangling** (`08_appendix/02_pandas_data_wrangling/`)  
  Series and DataFrames, BlockManager internals, missing data handling, and leakage-free transformations.

### Part 9: Model Inference & Interpretability
* **Unit 9.1: Model Inference and Interpretability** (`09_model_interpretability_and_inference/`)  
  Statistical inference in linear models (standard errors, $t$-statistics, $p$-values, confidence intervals, Cohen's $f^2$ effect sizes, odds ratios) and non-linear model explainability (Permutation Feature Importance, Partial Dependence Plots, ICE curves, LIME, and SHAP).

### Part 10: Deep Learning Foundations
* **Unit 10.1: Feedforward Neural Networks: Functional Approximation, Backpropagation, and Deep Training Dynamics** (`10_neural_networks/`)  
  Universal continuous functional approximation (Cybenko, Hornik), depth vs. width exponential space-folding, activation function mechanics (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, GELU), why symbolic differentiation fails, Backpropagation as reverse-mode automatic differentiation with a full step-by-step by-hand numerical walkthrough, regularization ($L_2$ weight decay, Dropout, Batch Normalization), learning rate annealing (Cosine, SGDR, warmup), batch size dynamics (flat vs. sharp minima, linear scaling rule, critical batch size), and deep training stability tricks.

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

Each folder contains a standalone `.tex` file with its corresponding compiled `.pdf`. If you modify any LaTeX notes, you can compile all files using the automated script:

```bash
python3 compile_all.py
```

To recompile an individual unit:

```bash
cd 03_linear_and_parametric_models/01_linear_regression/
pdflatex -interaction=nonstopmode linear_regression.tex
pdflatex -interaction=nonstopmode linear_regression.tex
```
*(Running twice ensures tables of contents and cross-references resolve properly).*
