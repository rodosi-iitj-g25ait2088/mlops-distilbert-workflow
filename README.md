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
