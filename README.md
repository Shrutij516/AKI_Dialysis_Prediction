# AKI Dialysis Risk Prediction

Predicting dialysis need (End-Stage Renal Disease, ESRD) within 2 years of an Acute Kidney Injury (AKI) episode using longitudinal EHR event sequences and deep learning.

**Course:** AMS 585 — Internship in Data Science, Stony Brook University  
**Supervisor:** Dr. Alisa Yurovsky  
**Period:** Spring 2026

---

## Problem Statement

Acute Kidney Injury is a sudden episode of kidney failure that is often reversible, but a significant subset of AKI survivors progress to permanent dialysis within two years. Early identification of these high-risk patients enables timely clinical intervention — nephrology referral, care planning, and preparation for renal replacement therapy.

This project frames the problem as **binary sequence classification**:

- **Input:** Chronological EHR event sequences (diagnosis, procedure, medication, and lab codes) recorded before the AKI episode
- **Output:** Will this patient require dialysis or reach ESRD within 2 years of AKI? (1 = yes, 0 = no)

---

## Dataset

**Source:** TriNetX `eGFR_gt20_95KPatients` — a de-identified EHR research dataset from a network of US healthcare organizations.

| Table | Description | Scale |
|---|---|---|
| `diagnosis.csv` | ICD-9 / ICD-10 diagnosis codes with dates | ~3.28M events |
| `procedure.csv` | CPT / ICD procedure codes with dates | ~2.61M events |
| `medication.csv` | Medication records with dates | ~2.89M events |
| `lab_result.csv` | Lab results with LOINC codes and numeric values | ~22M rows, 14.48 GB |

> Raw EHR data cannot be shared due to patient privacy and HIPAA restrictions.

### Cohort Construction

| Step | Detail |
|---|---|
| AKI identification | ICD-10: `N17*`, ICD-9: `584*` |
| Pre-existing ESRD exclusion | ICD-10: `N18.6`, ICD-9: `585.6` at or before AKI date |
| Final cohort | **31,399 AKI patients** |
| Positive outcome | ESRD (`N18.6`) or dialysis within 2 years of AKI |
| Dialysis codes | CPT: 90935, 90937, 90945, 90947 · HCPCS: G0257 · ICD-10-PCS: 5A1D · ICD-9: 39.95 |
| Positive class size | **3,191 patients (~10.2%)** |

### Temporal Windows

```
[--- all prior history ---][--- 90-day blank ---][AKI][--- 2-year prediction window ---]
         observation window       (excluded)     index         outcome ascertainment
```

- **Blanking window:** 90 days immediately before AKI index date — excluded from model inputs to prevent data leakage from the AKI episode itself
- **Observation window:** All clinical history before the blanking window
- **Prediction window:** 2 years after AKI for outcome labeling

---

## Modeling Approach

### Event Tokenization

Each clinical event is tokenized as a `type_code` pair (e.g., `diagnosis_N18.1`, `procedure_90937`, `lab_creatinine_severe`) and sorted chronologically into a patient sequence. Patients with fewer than 3 events are excluded, leaving **26,813 patients** with a vocabulary of **~60,000 unique tokens**. Sequences are capped at 400 events.

### Lab Data: Clinical Severity Binning

Rather than encoding raw LOINC codes (which discard the actual result), numeric lab values are binned into clinically grounded severity categories using standard guidelines (KDIGO staging for eGFR; standard ranges for creatinine and BUN):

| Lab | Bin example | Token |
|---|---|---|
| Creatinine = 4.5 mg/dL | Severe | `lab_creatinine_severe` |
| eGFR = 22 mL/min | G4 (severely reduced) | `lab_egfr_g4` |
| BUN elevated | High | `lab_bun_high` |

This produces ~13 clinically meaningful tokens instead of thousands of raw LOINC variants.

> **Bug fixed:** The original LOINC matching via `standardized_terminology.csv` mislabeled ~62% of eGFR measurements as creatinine (because eGFR formula descriptions contain the word "creatinine"). This was caught diagnostically — the median "creatinine" value in the mislabeled rows was 37 mg/dL, which is clinically impossible. Fixed by switching to a curated canonical LOINC code list.

### Models

Two sequence architectures are compared:

| Model | Description |
|---|---|
| **Transformer** | Multi-head self-attention; processes all events in parallel; captures long-range dependencies |
| **RETAIN** | REverse Time AttentIoN; GRU-based recurrent model that reads sequences in reverse chronological order; produces per-event attention weights for clinical interpretability |

Both models use **binary focal loss** (γ = 2, α tuned via grid search) to handle the ~10% positive rate. Hyperparameter search covers focal loss α ∈ {0.25, 0.5}, dropout ∈ {0.1, 0.2}, with early stopping on validation AUC.

---

## Results

4-way comparison across model × lab data inclusion (80/20 patient-level train/test split):

| Metric | Transformer No Labs | Transformer W/ Labs* | RETAIN No Labs | **RETAIN Binned Labs** |
|---|---|---|---|---|
| Accuracy | 0.775 | 0.649 | 0.664 | **0.875** |
| ROC-AUC | 0.631 | 0.625 | 0.638 | **0.653** |
| PR-AUC | 0.135 | 0.157 | 0.151 | **0.194** |
| Brier Score | **0.094** | **0.090** | 0.113 | 0.118 |
| F1 Score | 0.205 | 0.204 | 0.214 | **0.254** |
| Precision | 0.148 | 0.127 | 0.134 | **0.264** |
| Sensitivity | 0.336 | 0.513 | 0.529 | 0.244 |
| Specificity | 0.817 | 0.662 | 0.677 | **0.935** |

**RETAIN with binned labs achieves the best performance on 5 of 8 metrics.**

*\*Transformer W/ Labs uses the original naive LOINC encoding (before the misclassification bug fix) — this column is not directly comparable to RETAIN W/ Labs. See Limitations.*

Key finding: Adding lab data with raw LOINC codes hurt RETAIN (ROC-AUC dropped from 0.638 to 0.610) due to token dilution and sequence truncation. Clinical severity binning fully reversed this — RETAIN + binned labs reached ROC-AUC 0.653, PR-AUC 0.194, F1 0.254.

---

## Repository Structure

```
AKI_Dialysis_Prediction/
├── Transformer_NoLabs.ipynb     # Transformer model — diagnosis, procedure, medication only
├── Transformer_WithLabs.ipynb   # Transformer model — adds lab data (naive LOINC encoding)
├── RETAIN_NoLabs.ipynb          # RETAIN model — diagnosis, procedure, medication only
├── RETAIN_WithLabs.ipynb        # RETAIN model — adds lab data with clinical severity binning
└── README.md
```

---

## Limitations & Future Work

- **Incomparable lab columns:** Transformer+labs uses naive LOINC encoding; RETAIN+labs uses curated codes + severity binning. Applying the same encoding to the Transformer is the most direct next step.
- **Single train/test split:** No cross-validation; metrics may carry variance.
- **Limited lab panel:** Only creatinine, eGFR, BUN included. Albumin-to-creatinine ratio, potassium, and bicarbonate may add predictive signal.
- **No demographics:** Patient age, sex, and comorbidity burden were not incorporated.
- **Attention weights not extracted:** RETAIN's per-event attention weights were not yet validated; extracting them is the planned next step toward clinical explainability.
- **No probability calibration:** Predicted risk scores may not be reliable as absolute probabilities even when AUC is strong.
- **Generalizability:** Dataset covers a US healthcare network (TriNetX); results may not transfer to other populations or healthcare systems.
