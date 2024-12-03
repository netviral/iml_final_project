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

- **Basic Text Metrics**:
  - `total_words`: Total number of words in the text.
  - `unique_words`: Number of unique words in the text.
  - `total_characters`: Total number of characters in the text.
  - `total_sentences`: Total number of sentences in the text.
  - `total_paragraphs`: Total number of paragraphs in the text.
  - `average_word_length`: Average length of words in the text.

- **Complexity and Readability**:
  - `complex_word_density`: Ratio of complex words to total words.
  - `stopword_ratio`: Ratio of stopwords to total words.
  - `rare_word_ratio`: Ratio of rare words to total words.
  - `sentence_complexity`: Complexity score based on sentence structure.
  - `flesch_reading_ease`: Flesch Reading Ease score of the text.
  - `gunning_fog_index`: Gunning Fog Index for text readability.

- **Semantic Coherence and Redundancy**:
  - `cosine_similarity_redundancy`: Redundancy score based on cosine similarity.
  - `cosine_similarity_coherence`: Coherence score based on cosine similarity.

- **Sentence-Level Features**:
  - `sentence_length_variation`: Variation in sentence lengths.
  - `complexity_variation`: Variation in sentence complexity.
  - `avg_sentence_length`: Average length of sentences in the text.

- **Zipf's Law and Linguistic Diversity**:
  - `zipfs_law_adherence`: Measure of adherence to Zipf's law.
  - `pos_diversity`: Part-of-speech diversity in the text.

- **Grammatical Features**:
  - `noun_verb_ratio`: Ratio of nouns to verbs.
  - `adjective_adverb_ratio`: Ratio of adjectives to adverbs.
  - `grammar_error_density`: Density of grammatical errors in the text.
  - `passive_to_active_ratio`: Ratio of passive to active voice sentences.

- **Stylistic Features**:
  - `punctuation_density`: Density of punctuation marks.
  - `first_person_pronouns_density`: Density of first-person pronouns.
  - `second_person_pronouns_density`: Density of second-person pronouns.
  - `third_person_pronouns_density`: Density of third-person pronouns.

- **Repetition and Dependency Features**:
  - `repetition_frequency`: Frequency of word or phrase repetition.
  - `dependency_score`: Dependency parsing score for syntactic analysis.
  - `dependency_length`: Average dependency length in sentences.

- **Lexical and Discourse Features**:
  - `lexical_richness_MTLD`: Measure of lexical richness using the MTLD metric.
  - `discourse_markers_count`: Count of discourse markers in the text.
  - `modals_count`: Count of modal verbs in the text.
  - `epistemic_markers_count`: Count of epistemic markers.
  - `nominalisations_count`: Count of nominalizations.

- **Averages of Key Features**:
  - `avg_dependency_score`: Average dependency parsing score.
  - `avg_dependency_length`: Average dependency length in sentences.
  - `avg_discourse_markers`: Average number of discourse markers.
  - `avg_modals`: Average number of modal verbs.
  - `avg_epistemic_markers`: Average number of epistemic markers.
  - `avg_nominalisations`: Average number of nominalizations.

- **Advanced Linguistic Features**:
  - `function_word_density`: Density of function words in the text.
  - `avg_entropy`: Average entropy score for the text.
  - `avg_caesura`: Average caesura (pauses) in the text.

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
- Tools and libraries: `pandas
scikit-learn
joblib
flask
numpy
spacy
nltk
lexicalrichness
textblob`

# Additional Notes

## Installing spaCy Model
After installing `spaCy`, ensure you download the `en_core_web_sm` model using the following command:
```bash
python -m spacy download en_core_web_sm

Downloading NLTK Datasets
Some nltk functionalities require downloading additional datasets. Run the following commands in Python to download the necessary resources:

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('cmudict')

## References and Datasets

This project has utilized resources, datasets, and tools from the following sources:

1. **MeTA Toolkit**  
   Repository: [MeTA Toolkit GitHub](https://github.com/meta-toolkit/meta/tree/master)  
   Description: Used for reference on text analysis, including lexical diversity and linguistic features.

2. **Modals and Markers Study**  
   Repository: [ChatGPT Student Essay Study](https://github.com/sherbold/chatgpt-student-essay-study/blob/main/markers/modals.csv)  
   Description: Referenced for identifying linguistic markers such as modals in text analysis.

3. **Augmented Data for LLM Detection**  
   Dataset: [Augmented Data for LLM Detect AI-Generated Text](https://www.kaggle.com/datasets/jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text)  
   Description: A dataset of human-written and AI-generated text samples used for training and evaluation.

4. **DAIGT V2 Train Dataset**  
   Dataset: [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)  
   Description: Another dataset for detecting AI-generated text, helping enhance the model's robustness.

5. **LLM Detect AI-Generated Text Dataset**  
   Dataset: [LLM Detect Dataset](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset?resource=download)  
   Description: Used as a supplementary dataset for training and validating the machine learning model.

These resources and datasets were instrumental in the development of feature extraction methods, model training, and validation of the machine learning pipeline.
