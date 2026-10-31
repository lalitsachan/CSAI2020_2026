# Machine Learning: First Principles, Algorithms, and Practical Pipelines

A rigorous, end-to-end textbook and practical course guide on Machine Learning. Every algorithm is developed from fundamental mathematical principles—linear algebra, multivariate calculus, optimization, and probability—accompanied by complete step-by-step arithmetic derivations, vector geometry diagrams, and production-grade Python implementations.

---

## Pedagogical Philosophy

1. **No Magic Black Boxes:** We do not accept high-level API calls without understanding what happens underneath. Every loss function, decision boundary, split criterion, weight update, and projection is derived analytically before calling `scikit-learn` or any numerical library.
2. **Clear, Direct Language:** Mathematical rigor does not require archaic academic fluff. Complex ideas are explained directly, concretely, and pragmatically—connecting algorithmic trade-offs directly to business objectives, computational constraints, and risk management.
3. **From-Scratch Verification:** Every major family of algorithms (OLS, Ridge, Lasso, Logistic Regression, KNN, Naive Bayes, SVM, Decision Trees, Random Forests, Gradient Boosted Trees, XGBoost, CatBoost, K-Means, DBSCAN, Hierarchical Clustering, PCA) is implemented from scratch using NumPy and verified for exact numerical equivalence against Scikit-Learn.
4. **Vector Geometry via TikZ:** Every chapter features custom vector diagrams illustrating loss surfaces, hyperplanes, decision boundaries, dendrograms, and coordinate projections.

---

## Course Curriculum & Repository Structure

```
CSAI2020_2026/
├── 01_mathematical_foundations/
│   ├── 01_linear_algebra/             # Ch 01: Vector spaces, norms, spans, projections, eigendecomposition
│   └── 02_multivariate_calculus/       # Ch 02: Gradients, Hessians, Taylor series, steepest descent
├── 02_ml_fundamentals_and_pipelines/
│   ├── 01_anatomy_of_a_model/          # Ch 03: The 3 fundamental questions, loss functions (MSE, MAE, NLL)
│   ├── 02_loss_functions_and_metrics/  # Ch 04: Confusion matrix, Precision/Recall, F-beta, KS, ROC/AUC
│   ├── 03_data_processing_pipelines/   # Ch 05: Data cleaning pipelines [Notebook: data_preprocessing.ipynb]
│   └── 04_optimizers_in_depth/         # Ch 06: SGD, Momentum, Adam [Notebook: optimizers_in_depth.ipynb]
├── 03_linear_and_parametric_models/
│   ├── 01_linear_regression/           # Ch 07: Normal equations & BLUE [Notebook: linear_regression.ipynb]
│   ├── 02_regularization_l1_l2/        # Ch 08: Ridge L2 & Lasso L1 geometry [Notebook: regularization_l1_l2.ipynb]
│   └── 03_logistic_regression/         # Ch 09: Sigmoid link & KS cutoffs [Notebook: logistic_regression.ipynb]
├── 04_classical_and_instance_models/
│   ├── 01_k_nearest_neighbors/         # Ch 10: Minkowski spaces & scaling [Notebook: k_nearest_neighbors.ipynb]
│   ├── 02_naive_bayes/                 # Ch 11: Text classification with TF-IDF [Notebook: naive_bayes.ipynb]
│   └── 03_support_vector_machines/     # Ch 12: Margin maximization & RBF kernel [Notebook: support_vector_machines.ipynb]
├── 05_tree_based_and_ensemble_models/
│   ├── 01_decision_trees/              # Ch 13: Recursive partitioning & Gini [Notebook: decision_trees.ipynb]
│   ├── 02_bagging_rf_extra_trees/      # Ch 14: Bagging, RF & Extra Trees [Notebook: random_forest_extra_trees.ipynb]
│   └── 03_hyperparameter_tuning/       # Ch 15: Grid vs Random Search [Notebook: hyperparameter_tuning.ipynb]
├── 06_boosting_machines/
│   ├── 01_gradient_boosting_machines/  # Ch 16: Sequential error correction & shrinkage
│   ├── 02_xgboost/                     # Ch 17: 2nd-order Taylor objective & gain [Notebook: xgboost.ipynb]
│   └── 03_catboost/                    # Ch 18: Ordered Target Statistics & oblivious trees
├── 07_unsupervised_learning/
│   ├── 01_proximity_and_hierarchical/  # Ch 19: Distance matrices & dendrograms [Notebook: hierarchical_clustering.ipynb]
│   ├── 02_kmeans_and_pam/              # Ch 20: K-Means, K-Medoids & Silhouette [Notebook: kmeans_and_pam.ipynb]
│   ├── 03_dbscan/                      # Ch 21: Density reachability & noise [Notebook: dbscan.ipynb]
│   └── 04_pca/                         # Ch 22: Covariance eigendecomposition & SVD [Notebook: pca.ipynb]
├── 08_appendix/
│   ├── 01_numpy_crash_course/          # App A: Strides, SIMD & broadcasting [Notebook: numpy_crash_course.ipynb]
│   └── 02_pandas_data_wrangling/       # App B: Columnar storage & groupby [Notebook: pandas_data_wrangling.ipynb]
├── data/                               # Centralized real-world datasets
│   ├── bd_train.csv & bd_test.csv      # Bank deposit subscription campaign
│   ├── loan_data_train.csv & test.csv  # Peer-to-peer credit risk and interest rates
│   ├── insurance.csv                   # Healthcare expenditure regression
│   ├── diabetic_data.csv               # Hospital readmission classification
│   └── Consumer_Complaints_train.csv.zip # Financial complaint text classification
└── utils/
    └── ml_utils.py                     # Production-grade preprocessing & threshold tuning pipelines
```

---

## Datasets Catalog

All datasets are stored in `data/` and used across chapters:
* **`loan_data_train.csv` / `loan_data_test.csv`**: Peer-to-peer lending records. Used in Linear Regression, Regularization, and Decision Trees to predict interest rates and default risk.
* **`bd_train.csv` / `bd_test.csv`**: Direct marketing campaign for term deposits. Used in Logistic Regression, KS cutoff tuning, ROC/AUC analysis, and Ensemble modeling.
* **`insurance.csv`**: US health insurance charges. Used for non-linear regression, homoscedasticity diagnostics, and feature transformations.
* **`diabetic_data.csv`**: Multi-hospital clinical records of diabetic patients. Used in high-cardinality categorical encoding, missing value strategies, and Tree Boosting.
* **`Consumer_Complaints_train.csv.zip`**: Real-world CFPB consumer financial complaints. Used for text processing, bag-of-words, TF-IDF, and Naive Bayes / Linear SVM text classification.

---

## Standard Mathematical Notation

To avoid ambiguity, notation is held strictly consistent across all chapters:

| Symbol | Mathematical Meaning | Python / NumPy Representation |
| :--- | :--- | :--- |
| $\mathbf{X} \in \mathbb{R}^{n \times p}$ | Design matrix ($n$ observations, $p$ features) | `X` (`ndarray` or `DataFrame`) |
| $\mathbf{x}_i \in \mathbb{R}^p$ | Column vector representing sample $i$ | `X[i, :]` |
| $\mathbf{y} \in \mathbb{R}^n$ | Target vector | `y` |
| $\mathbf{w} \in \mathbb{R}^p, b \in \mathbb{R}$ | Model weights and intercept / bias | `w`, `b` |
| $\hat{y}_i = f(\mathbf{x}_i; \mathbf{w})$ | Model prediction for sample $i$ | `y_pred` |
| $\mathcal{L}(\mathbf{w})$ | Objective / Loss function | `loss` |
| $\nabla \mathcal{L}(\mathbf{w})$ | Gradient vector with respect to $\mathbf{w}$ | `grad` |
| $\mathbf{H} = \nabla^2 \mathcal{L}(\mathbf{w})$ | Symmetric Hessian matrix of second derivatives | `hessian` |
| $\eta > 0$ | Learning rate / step size parameter | `learning_rate` or `eta` |
| $\mathbf{\Sigma} \in \mathbb{R}^{p \times p}$ | Sample covariance matrix | `np.cov(X.T)` |
| $\mathbf{u}_j, \lambda_j$ | Eigenvectors and corresponding eigenvalues | `eigenvectors`, `eigenvalues` |

---

## Compiling the LaTeX Chapters

Every chapter is a standalone LaTeX document configured with standard packages (`amsmath`, `booktabs`, `listings`, `tikz`, `hyperref`). 

To compile any individual chapter into a PDF:
```bash
cd 03_linear_and_parametric_models/01_linear_regression/
pdflatex -interaction=nonstopmode linear_regression.tex
pdflatex -interaction=nonstopmode linear_regression.tex  # Second run resolves TOC and cross-references
```

To compile all 22 chapters and appendices in batch, run the included build script:
```bash
python3 compile_all.py
```
This script compiles each chapter, resolves cross-references and tables of contents, and leaves clean `.pdf` files in each chapter directory while respecting `.gitignore` rules for auxiliary files.

