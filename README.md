# MLOps Assignment 2: Text Classification Pipeline
## Genre Categorization for Goodreads Reviews via Fine-Tuned DistilBERT

This project details an end-to-end MLOps workflow designed to classify Goodreads book reviews into seven distinct genres using a fine-tuned `distilbert-base-cased` architecture. The implementation showcases standard MLOps practices, including GPU-accelerated training via Kaggle, experiment logging with Weights & Biases, and model hosting on the Hugging Face Hub.

---

## Project Framework

- [Project Objective](#project-objective)
- [Dataset Details](#dataset-details)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Design](#model-design)
- [Training Parameters](#training-parameters)
- [MLOps Integration](#mlops-integration)
- [Performance Metrics](#performance-metrics)
- [Repository Layout](#repository-layout)
- [Execution & Reproduction](#execution--reproduction)
- [Project Artifacts](#project-artifacts)

---

## Project Objective

The core task is multi-class text classification: analyzing unstructured text from Goodreads reviews to determine the corresponding book genre. The model categorizes inputs into one of seven target genres:

*   Poetry
*   Comics & Graphic
*   Fantasy & Paranormal
*   History & Biography
*   Mystery, Thriller & Crime
*   Romance
*   Young Adult

> **Note:** The primary aim of this assignment is to establish a robust, tracked, and easily deployable machine learning engineering pipeline rather than optimizing raw predictive accuracy.

---

## Dataset

*   **Origin:** Data is sourced from the UCSD Goodreads Book Graph, utilizing individual genre files in `.json.gz` format.
*   **Sampling Strategy:** Subsets of reviews were extracted across all genres to accommodate training constraints on free tier hardware.
*   **Data Partitioning:** A stratified 80/20 split was applied to separate training and evaluation sets while maintaining balanced class representations.
*   **Features:** The `review_text` column serves as the input sequence, capped at a maximum length of 512 tokens during tokenization.

---

## Pipeline Architecture

Raw Goodreads Reviews (7 genres, UCSD Book Graph)
        │
        ▼
  Data Loading & Sampling
        │
        ▼
  DistilBERT Tokenization
  (distilbert-base-cased, max_length=512)
        │
        ▼
  Fine-Tuning on Kaggle GPU (T4)
  (HuggingFace Trainer API, 3 epochs)
        │
        ├──► Experiment Tracking (Weights & Biases)
        │         - Loss, Accuracy, F1 per epoch
        │         - Hyperparameters logged
        │         - Eval report as W&B Artifact
        ▼
  Evaluation on Test Set
  (Accuracy, F1, Classification Report)
        │
        ▼
  Model Deployment → Hugging Face Hub
  (publicly accessible for inference)


---

## Model Architecture

| Component        | Detail                                  |
|------------------|-----------------------------------------|
| Base model       | `distilbert-base-cased`                 |
| Task head        | `DistilBertForSequenceClassification`   |
| Number of labels | 7                                       |
| Max token length | 512                                     |
| Framework        | HuggingFace Transformers + PyTorch      |

**Why DistilBERT?**
DistilBERT is a distilled (compressed) version of BERT that retains ~97% of BERT's language understanding while being 40% smaller and 60% faster. For an MLOps assignment focused on workflow rather than state-of-the-art accuracy, DistilBERT offers the ideal balance between performance and training speed on free Kaggle GPU resources. Its well-documented HuggingFace integration also makes it straightforward to load, fine-tune, and deploy within a single pipeline.

---

## Training Configuration

| Hyperparameter       | Value                        |
|----------------------|------------------------------|
| Epochs               | 3                            |
| Train batch size     | 16                           |
| Eval batch size      | 32                           |
| Learning rate        | 3e-5                         |
| Warmup steps         | 100                          |
| Weight decay         | 0.01                         |
| Logging steps        | 50                           |
| Evaluation strategy  | Every epoch                  |
| Optimizer            | AdamW (HuggingFace default)  |
| Training platform    | Kaggle Notebook (GPU T4)     |

---

## MLOps Components

### 1. Experiment Tracking — Weights & Biases
- All runs logged automatically via `report_to="wandb"` in `TrainingArguments`
- Tracks: training loss, eval loss, accuracy, F1 score per epoch
- Hyperparameters captured in W&B config for full reproducibility
- Final evaluation report saved as a versioned **W&B Artifact**

### 2. Training Platform — Kaggle Notebooks
- Hardware: **GPU T4** (free tier, 30 hrs/week)
- Internet enabled for HuggingFace model downloads and Hub push
- API credentials stored securely via **Kaggle Secrets** (`WANDB_API_KEY`, `HF_TOKEN`)
- Zero hardcoded credentials in any code

### 3. Model Deployment — Hugging Face Hub
- Fine-tuned model and tokenizer pushed to HuggingFace Hub
- Publicly accessible — anyone can load and run inference with one line:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="DishaSinghania/distilbert-goodreads-genres")
result = classifier("This fantasy novel had incredible world-building and magic systems.")
print(result)
```

## Results
MetricScoreAccuracy  |  0.6013  |
Weighted F1          |  0.5979  |
Eval Loss            |  2.2483  |

---

### What the W&B Charts Showed
- Training loss decreased steadily across all 3 epochs — model was learning
- Validation loss tracked training loss closely — no significant overfitting
- Accuracy and F1 improved each epoch, confirming the fine-tuning was effective

---

## Project Structure

```
mlops-assignment2/
├── mlops-assignment.ipynb      # Kaggle notebook — full training pipeline
├── train_pipeline.py           # Python script exported from Kaggle notebook
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Setup & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/Disha19/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```
> On Kaggle: use **Add-ons → Secrets** to store these securely instead of environment variables.

### 4. Run on Kaggle (Recommended)
- Import `mlops-assignment.ipynb` into Kaggle
- Enable GPU: Settings → Accelerator → GPU T4
- Enable Internet: Settings → Environment Preferences → Internet ON
- Add secrets: Add-ons → Secrets → add `WANDB_API_KEY` and `HF_TOKEN`
- Click **Run All**

### 5. Run locally (CPU only — slow)
```bash
python train_pipeline.py
```
> For CPU runs, reduce sample size to 200 per genre and set `device = 'cpu'`

---

## Links

| Resource            | URL                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------|
|  Hugging Face Model | [Rodosi/mlops-distilbert-workflow](https://huggingface.co/Rodosi/mlops-distilbert-workflow) |
|  W&B Dashboard      | [mlops-assignment2-rodosi](https://wandb.ai/g25ait2088-iit-jodhpur/mlops-assignment2-rodosi)|
|  Kaggle Notebook    | [mlops-assignment2](https://www.kaggle.com/code/rodosibiswas/mlops-assignment2)             |
|  GitHub Repository  | [mlops-assignment2](https://github.com/rodosi-iitj-g25ait2088/mlops-distilbert-workflow)    |
