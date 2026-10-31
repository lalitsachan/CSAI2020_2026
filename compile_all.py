#!/usr/bin/env python3
"""
compile_all.py
Automated compilation script for all LaTeX chapters and appendices in ml_class_book.
Runs pdflatex twice per document to resolve cross-references and tables of contents,
then cleans up temporary build artifacts according to .gitignore.
"""

import os
import subprocess
import sys
from pathlib import Path

# List of all tex files in their proper course sequence
CHAPTER_FILES = [
    # Part 1: Mathematical Foundations
    "01_mathematical_foundations/01_linear_algebra/linear_algebra_for_ml.tex",
    "01_mathematical_foundations/02_multivariate_calculus/multivariate_calculus_for_ml.tex",
    
    # Part 2: ML Fundamentals & Pipelines
    "02_ml_fundamentals_and_pipelines/01_anatomy_of_a_model/anatomy_of_a_model.tex",
    "02_ml_fundamentals_and_pipelines/02_loss_functions_and_metrics/loss_functions_and_metrics.tex",
    "02_ml_fundamentals_and_pipelines/03_data_processing_pipelines/data_processing_pipelines.tex",
    "02_ml_fundamentals_and_pipelines/04_optimizers_in_depth/optimizers_in_depth.tex",
    
    # Part 3: Linear & Parametric Models
    "03_linear_and_parametric_models/01_linear_regression/linear_regression.tex",
    "03_linear_and_parametric_models/02_regularization_l1_l2/regularization_l1_l2.tex",
    "03_linear_and_parametric_models/03_logistic_regression/logistic_regression.tex",
    
    # Part 4: Classical & Instance-Based Models
    "04_classical_and_instance_models/01_k_nearest_neighbors/k_nearest_neighbors.tex",
    "04_classical_and_instance_models/02_naive_bayes/naive_bayes.tex",
    "04_classical_and_instance_models/03_support_vector_machines/support_vector_machines.tex",
    
    # Part 5: Tree-Based & Ensemble Models
    "05_tree_based_and_ensemble_models/01_decision_trees/decision_trees.tex",
    "05_tree_based_and_ensemble_models/02_bagging_rf_extra_trees/bagging_rf_extra_trees.tex",
    "05_tree_based_and_ensemble_models/03_hyperparameter_tuning/hyperparameter_tuning.tex",
    
    # Part 6: Boosting Machines
    "06_boosting_machines/01_gradient_boosting_machines/gradient_boosting_machines.tex",
    "06_boosting_machines/02_xgboost/xgboost.tex",
    "06_boosting_machines/03_catboost/catboost.tex",
    
    # Part 7: Unsupervised Learning
    "07_unsupervised_learning/01_proximity_and_hierarchical/proximity_and_hierarchical.tex",
    "07_unsupervised_learning/02_kmeans_and_pam/kmeans_and_pam.tex",
    "07_unsupervised_learning/03_dbscan/dbscan.tex",
    "07_unsupervised_learning/04_pca/pca.tex",
    
    # Part 8: Appendices
    "08_appendix/01_numpy_crash_course/numpy_crash_course.tex",
    "08_appendix/02_pandas_data_wrangling/pandas_data_wrangling.tex",
]

def compile_document(rel_path_str: str, root_dir: Path) -> bool:
    tex_path = root_dir / rel_path_str
    if not tex_path.exists():
        print(f"[ERROR] File not found: {tex_path}")
        return False
        
    doc_dir = tex_path.parent
    doc_name = tex_path.name
    pdf_name = tex_path.stem + ".pdf"
    
    print(f"\n========================================================")
    print(f"Compiling: {rel_path_str}")
    print(f"Directory: {doc_dir}")
    print(f"========================================================")
    
    pdflatex_bin = "/Library/TeX/texbin/pdflatex"
    if not os.path.exists(pdflatex_bin):
        pdflatex_bin = "pdflatex"
        
    cmd = [pdflatex_bin, "-interaction=nonstopmode", doc_name]
    
    # First pass
    proc1 = subprocess.run(cmd, cwd=doc_dir, capture_output=True, text=True)
    if proc1.returncode != 0:
        print(f"[FAIL] Pass 1 failed for {doc_name}")
        # Print relevant error lines from log
        log_file = doc_dir / (tex_path.stem + ".log")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line, end="")
        return False
        
    # Second pass for Table of Contents, References, and Page Numbers
    proc2 = subprocess.run(cmd, cwd=doc_dir, capture_output=True, text=True)
    if proc2.returncode != 0:
        print(f"[FAIL] Pass 2 failed for {doc_name}")
        return False
        
    pdf_path = doc_dir / pdf_name
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"[SUCCESS] Generated: {pdf_name} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[FAIL] Output PDF not created: {pdf_path}")
        return False

def main():
    root_dir = Path(__file__).resolve().parent
    print(f"Root Directory: {root_dir}")
    print(f"Total documents to compile: {len(CHAPTER_FILES)}")
    
    results = {}
    for rel_path in CHAPTER_FILES:
        success = compile_document(rel_path, root_dir)
        results[rel_path] = success
        
    print("\n" + "="*60)
    print("COMPILATION SUMMARY")
    print("="*60)
    all_passed = True
    for rel_path, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"[{status:6s}] {rel_path}")
        if not success:
            all_passed = False
            
    if all_passed:
        print("\nAll 24 documents (22 chapters + 2 appendices) compiled successfully!")
        sys.exit(0)
    else:
        print("\nSome documents failed compilation. Review the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

