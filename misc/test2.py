import re
import ssl
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string
import os
import csv
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from helper_functions import calculate_dep_score, calculate_dep_length, calculate_lex_richness_MTLD2, count_discourse_markers, count_total_modals, find_epistemic_markers, nominalisation_counter, average_per_sentence, process_text

# Load the CSV file
df = pd.read_csv('new_small_data.csv', delimiter=',')


# Load CMU Pronouncing Dictionary


def process_csv(input_file, output_file, log_file):
    # Check if the output file exists, otherwise initialize it with headers
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # writer.writerow(["text", "generated"] + list(process_text("").keys()))  # Include your custom keys

    # Get the last processed row from the log file
    start_row = 0
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            start_row = int(f.read().strip())

    # Read the input file in chunks
    for chunk in pd.read_csv(input_file, chunksize=1, skiprows=range(1, start_row + 1), encoding='utf-8', iterator=True):
        row = chunk.iloc[0]
        text = row["text"]
        generated = row["generated"]

        # Process the text
        processed_data = process_text(text)
        
        # Create a row for the output file
        output_row = [text, generated] + list(processed_data.values())

        # Write the processed row to the output file
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_row)

        # Update the log file
        start_row += 1
        with open(log_file, 'w') as f:
            f.write(str(start_row))

        print(f"Processed row {start_row}")  # Log progress

# File paths
input_file = "cleaned_training_human.csv"
output_file = "processed_training_human.csv"
log_file = "progress.log"

# Run the processing function
# process_csv(input_file, output_file, log_file)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Load CSV
csv_file = "metrics_with_labels.csv"  # Replace with your file name
data = pd.read_csv(csv_file)

# Prepare features and labels
feature_columns = [
     "average_word_length", "complex_word_count",
    "stopword_ratio", "rare_word_ratio", "lexical_diversity",
    "sentence_complexity", "flesch_reading_ease", "gunning_fog_index",
    "cosine_similarity_redundancy", "cosine_similarity_coherence",
    "sentence_length_variation", "complexity_variation", "avg_sentence_length",
    "zipfs_law_adherence", "pos_diversity", "noun_verb_ratio",
    "adjective_adverb_ratio", "grammar_error_density",
    "passive_to_active_ratio", "punctuation_density",
    "first_person_pronouns_density", "second_person_pronouns_density",
    "third_person_pronouns_density", "repetition_frequency"
]
X = data[feature_columns]
y = data['label']  # Replace 'label' with your actual label column

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Output coefficients
coefficients = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
print("\nLogistic Regression Coefficients:")
print(coefficients)


def predict_new_text(text, model, feature_columns):
    """
    Process a new piece of text, calculate its metrics, and predict if it's AI-generated.
    """
    # Calculate metrics for the new text
    metrics = process_text(text)  # Replace with your actual implementation
    metrics_values = [metrics[col] for col in feature_columns]
    
    # Predict the probability
    prediction_probabilities = model.predict_proba([metrics_values])[0]
    prediction = model.predict([metrics_values])[0]
    
    return {
        "prediction": prediction,  # 0 or 1
        "probability_AI_generated": prediction_probabilities[1],
        "probability_human_written": prediction_probabilities[0],
    }

# Example usage with a new text
new_text = """
"""
test_result = predict_new_text(new_text, model, feature_columns)

print("Test Results for New Text:")
print(f"Prediction: {'AI-Generated' if test_result['prediction'] == 1 else 'Human-Written'}")
print(f"Probability AI-Generated: {test_result['probability_AI_generated']:.2f}")
print(f"Probability Human-Written: {test_result['probability_human_written']:.2f}")