# AI Text Classification Project

## Overview
This project uses a pre-trained Machine Learning model to classify text as **AI-generated** or **human-written**. The model leverages linguistic features such as discourse markers and phonetic data to achieve accurate predictions. The project includes both data processing scripts and a lightweight web application to demonstrate the functionality.

---

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Machine Learning Details](#machine-learning-details)
- [Data Pipeline](#data-pipeline)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Text Analysis**: Uses natural language processing (NLP) techniques to extract meaningful features from text.
- **Machine Learning**: A pre-trained logistic regression model performs predictions.
- **Scalable Design**: Modular components for easy extension to other ML tasks.
- **Web Interface**: A simple Node.js and HTML-based app for demonstration purposes.

---

## Directory Structure

```plaintext
.
├── app/                  # Node.js server files
├── cmudict/              # Pronunciation dictionary for text analysis
├── markers/              # Files containing discourse markers
├── misc/                 # Miscellaneous utilities
├── model/                # Pre-trained ML model and feature columns
├── processed_data/       # Pre-processed data files
├── raw_data/             # Raw data files for analysis
├── clean_csv_file.py     # Script to clean raw CSV files
├── helper_functions.py   # Helper functions for feature extraction
├── main_calculate_feature_values.py # Script to calculate features from raw text
├── predict.py            # Python script to load model and predict
├── progress.log          # Log file for development tracking
```

---

## Setup Instructions

### Prerequisites

- **Python** (3.10 or above)
- **Pip** to install dependencies
- **Node.js** (only if running the web app)

### Python Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run `predict.py` to test the prediction script:
   ```bash
   python3 predict.py
   ```

---

## Machine Learning Details

### Model

- **Type**: Logistic Regression
- **Framework**: `scikit-learn`
- **Training Data**:
  - Extracted features from a dataset of AI-generated and human-written text.
  - Features include:
    - Discourse markers (e.g., conjunctions, transitions).
    - Phonetic analysis using CMU Pronouncing Dictionary (`cmudict/`).
    - Statistical text properties (e.g., word frequency, sentence length).
- **Saved Model**: The trained model and feature column mappings are stored in the `model/` directory.

### Features

The ML model uses custom-engineered features:
1. **Discourse Markers**:
   - Captures linguistic structures using a predefined list in `markers/`.
2. **Phonetic Features**:
   - Uses the CMU Pronouncing Dictionary (`cmudict/`) for phoneme-level analysis.
3. **Statistical Metrics**:
   - Word and sentence lengths, frequency analysis, and more.

### Prediction Process

- The `predict.py` script:
  1. Reads the input text.
  2. Processes it to extract features using `helper_functions.py`.
  3. Loads the logistic regression model.
  4. Outputs the prediction as a JSON object.

---

## Data Pipeline

### Raw Data

- Raw datasets are stored in `raw_data/`.
- Example: AI-generated text from GPT models and human-written text from curated sources.

### Pre-Processing

- The `clean_csv_file.py` script cleans raw data.
- The `main_calculate_feature_values.py` script calculates features based on the cleaned data.

### Processed Data

- Outputs of feature extraction are saved in `processed_data/` for reusability.

---

## Usage

1. **Run the Predictor**:
   - Use `predict.py` to make standalone predictions:
     ```bash
     python3 predict.py
     ```
   - Input text via stdin and view JSON output with prediction probabilities.

2. **Optional Web Interface**:
   - Navigate to `http://localhost:3000` if the Node.js server is running.

---

## Acknowledgments

- Developed by **Ibrahim Khalil**, integrating linguistic feature extraction and machine learning techniques.
- Tools and libraries: `scikit-learn`, `pandas`, `numpy`, `NLTK`, `Node
