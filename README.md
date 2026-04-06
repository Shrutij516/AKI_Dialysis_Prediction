# AKI Dialysis Risk Prediction using Machine Learning

## Overview

Acute Kidney Injury (AKI) is a critical clinical condition that can progress to kidney failure requiring dialysis. Early identification of high-risk patients is essential for improving patient outcomes and enabling timely medical intervention.

This project develops machine learning models to predict dialysis requirement in AKI patients using structured Electronic Health Record (EHR) data, including temporal clinical events.

The work is conducted as part of a course-based research project at Stony Brook University.

## Problem Statement

Given a patient diagnosed with Acute Kidney Injury (AKI), predict whether the patient will progress to dialysis or end-stage renal disease (ESRD) within a future time window.

This task is framed as a binary classification problem:
- Input: Structured EHR features
- Output: Dialysis required (1) / No dialysis required (0)

## Dataset

The dataset consists of structured EHR variables, including:

- Diagnosis codes (ICD)
- Procedure codes (CPT / ICD)
- Medication records
- Laboratory data (planned extension)
- Patient timeline events

1. Cohort Definition
   - AKI Identified using:
          - ICD-10: N17*
          - ICD-9: 584*
   - Index date = first AKI diagnosis
   - Patients with prior ESRD removed:
          - ICD-10: N18.6
          - ICD-9: 585.6
3. Outcome Definition
   - Prediction window: 2 years after AKI
   - Positive class:
          - ESRD OR dialysis
   - Final prevalence: ~10% positive cases
5. Scale of Data
   - Diagnosis events: ~3.28M
   - Procedure events: ~2.61M
   - Medication events: ~2.89M
   - Total events processed: ~8.7 million
   - Vocabulary size: ~60K unique tokens

This large-scale longitudinal dataset enables modeling of complex temporal clinical patterns.

Due to patient privacy and HIPAA restrictions, raw EHR data cannot be shared.

## Machine Learning Pipeline

The project follows a temporal ML pipeline tailored for clinical sequence modeling.

## 1. Data Preprocessing

Preprocessing steps include:

- Cohort construction and leakage prevention
- Observation window design:
       - 1-year and 90-day variants
- Alignment of events within observation window:
       - obs_start_date ≤ event_date < index_date
- Removal of sparse patients (e.g., <3 events)
- Fixing ingestion/alignment issues in raw data


## 2. Feature Engineering

### Multi-Modal Event Integration
Combined multiple EHR modalities:
- Diagnoses
- Procedures
- Medications

### Sequence Construction
Patient data transformed into chronological sequences:
Example:
[Diagnosis] → [Medication] → [Procedure]
- Total events processed: ~8.7M
- Sequence length capped at 400

This enables modeling of temporal clinical progression.


## 3. Handling Class Imbalance 

In clinical datasets, dialysis events are relatively rare, resulting in strong class imbalance.

Example:
- Dialysis required: minority class
- No dialysis: majority class

This imbalance can cause models to bias toward predicting the majority class.

### Mitigation strategies explored:
- SMOTE (Synthetic Minority Oversampling Technique)
- Class weighting
- Threshold tuning
- Evaluation with recall-sensitive metrics

Handling class imbalance is critical for clinical risk prediction tasks.

## 4. Models training

### Transformer-Based Model
A self-attention based architecture was used to capture long-range dependencies in patient history.

Key components:
- Token + positional embeddings
- Multi-head self-attention
- Feedforward layers
- Global pooling for classification

### Hyperparameters
- Embedding dimension: 128 (selected over 256)
- Number of heads: 4
- Max sequence length: 400

### Loss Functions
- Focal Loss (used)
- Used to address class imbalance:
       - Alpha ∈ {0.25, 0.5, 0.75}
       - Gamma = 2.0

This forces the model to focus on hard-to-predict minority class samples.

## 5. Model Evaluation
- Model performance is evaluated using clinically relevant metrics:

### ROC-AUC
Measures discrimination ability.

### F1 Score
Balances precision and recall.

### Precision / Recall
Precision: reliability of positive predictions
Recall: ability to detect high-risk patients


## 6. Hyperparameter Optimization
### Threshold Tuning
- Searched thresholds from 0.05 to 0.95 (step 0.01)
- Selected threshold maximizing F1 score

### Alpha Tuning
- Tested α ∈ {0.25, 0.5, 0.75}

### Hidden Dimension Tuning
- Compared 128 vs 256
- Selected 128 for better generalization

### Grid Search
- α ∈ {0.25, 0.5}
- Dropout ∈ {0.1, 0.2}
- Best config selected automatically based on F1


## 7. Results
### 1-Year Observation Window
- F1 ≈ 0.28
- ROC-AUC ≈ 0.68

### 90-Day Observation Window
- F1 ≈ 0.23
- ROC-AUC ≈ 0.66

### Comparison Insight
- Longer observation window improves performance
- Captures chronic disease progression patterns


## Experimental Workflow

The workflow consists of:

1. Data Cleaning  
2. Feature Engineering  
3. Model Training  
4. Hyperparameter Tuning  
5. Model Evaluation  
6. Clinical Risk Analysis
