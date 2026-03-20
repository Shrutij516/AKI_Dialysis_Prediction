AKI Dialysis Risk Prediction using Machine Learning

Overview

Acute Kidney Injury (AKI) is a critical clinical condition that can progress to kidney failure requiring dialysis. Early identification of high-risk patients is essential for improving patient outcomes and enabling timely medical intervention.

This project develops machine learning models to predict dialysis requirement in AKI patients using structured Electronic Health Record (EHR) features.

The work is conducted as part of a course-based research project at Stony Brook University.

Problem Statement

Given a patient admitted with Acute Kidney Injury, predict whether the patient will require dialysis during the hospital stay using structured clinical features extracted from Electronic Health Records.

This task is framed as a binary classification problem:

Input: Structured EHR features
Output: Dialysis required (1) / No dialysis required (0)
Dataset

The dataset consists of structured EHR variables, including:

Patient Demographics

Age

Sex

Admission characteristics

Laboratory Variables

Serum creatinine

Blood urea nitrogen (BUN)

Electrolytes

Other kidney function markers

Comorbidities

Hypertension

Diabetes

Chronic kidney disease history

Clinical Measurements

Vitals

Clinical lab results

Hospital encounter variables

Due to patient privacy and HIPAA restrictions, raw EHR data cannot be shared.

Machine Learning Pipeline

The project follows a standard ML pipeline for clinical prediction tasks.

1. Data Preprocessing

Preprocessing steps include:

Handling missing clinical values

Median / domain-informed imputation

Feature normalization and scaling

Encoding categorical variables

Filtering rare or noisy variables

Clinical datasets often contain substantial missingness and require careful preprocessing.

Feature Engineering

Features are derived from multiple clinical domains:

Laboratory trends

Demographic indicators

Comorbidity flags

Derived clinical risk markers

Feature engineering helps improve model interpretability and predictive power.

Class Imbalance Challenge

In clinical datasets, dialysis events are relatively rare, resulting in strong class imbalance.

Example:

Dialysis required: minority class
No dialysis: majority class

This imbalance can cause models to bias toward predicting the majority class.

Mitigation strategies explored:

SMOTE (Synthetic Minority Oversampling Technique)

Class weighting

Threshold tuning

Evaluation with recall-sensitive metrics

Handling class imbalance is critical for clinical risk prediction tasks.

Models Explored

Several baseline models were implemented to compare predictive performance.

Logistic Regression

Used as an interpretable baseline model for clinical prediction.

Random Forest

Captures nonlinear relationships between clinical variables.

Gradient Boosting

Improves predictive performance through ensemble learning.

Sequence-Based Extensions (Future Work)

Clinical data can contain temporal sequences of lab values and vitals.

Future extensions of this work include:

Recurrent neural networks (LSTM / GRU)

Transformer-based architectures for sequential EHR modeling

Temporal feature embeddings

Sequence models may better capture progression of kidney injury over time.

Loss Functions

Standard binary classification loss was used:

Binary Cross-Entropy Loss
L = -[y log(p) + (1-y) log(1-p)]

Where:

𝑦
y = true label

𝑝
p = predicted probability

When handling class imbalance, weighted versions of cross-entropy can be used.

Model Evaluation

Model performance is evaluated using metrics relevant to clinical decision-support systems.

ROC-AUC

Measures ability of the model to discriminate between dialysis and non-dialysis patients.

F1 Score

Balances precision and recall.

Sensitivity (Recall)

Critical in healthcare settings to minimize missed high-risk patients.

Sensitivity = True Positives / (True Positives + False Negatives)

High sensitivity is important to ensure high-risk patients are identified.

Experimental Workflow

The workflow consists of:

Data Cleaning
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Clinical Risk Analysis
