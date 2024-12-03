import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import joblib  # For saving and loading the model

# Load CSV
csv_file = "../processed_data/Final_Training_Done.csv"  # Replace with your file name
data = pd.read_csv(csv_file)
# print(data.columns)
data.columns = data.columns.str.strip()

# Prepare features and labels
# feature_columns = [
#     "total_words", "unique_words", "total_characters", "total_sentences",
#     "total_paragraphs", "average_word_length", "complex_word_count",
#     "stopword_ratio", "rare_word_ratio", "lexical_diversity",
#     "sentence_complexity", "flesch_reading_ease", "gunning_fog_index",
#     "cosine_similarity_redundancy", "cosine_similarity_coherence",
#     "sentence_length_variation", "complexity_variation", "avg_sentence_length",
#     "zipfs_law_adherence", "pos_diversity", "noun_verb_ratio",
#     "adjective_adverb_ratio", "grammar_error_density",
#     "passive_to_active_ratio", "punctuation_density",
#     "first_person_pronouns_density", "second_person_pronouns_density",
#     "third_person_pronouns_density", "repetition_frequency"
# ]

feature_columns=[
    "average_word_length", 
    "complex_word_density", "stopword_ratio", "rare_word_ratio", "sentence_complexity", 
    "flesch_reading_ease", "gunning_fog_index", "cosine_similarity_redundancy", "cosine_similarity_coherence", 
    "sentence_length_variation", "complexity_variation", "avg_sentence_length", 
    "zipfs_law_adherence", "pos_diversity", "noun_verb_ratio", "adjective_adverb_ratio", 
    "grammar_error_density", "passive_to_active_ratio", 
    "punctuation_density", "first_person_pronouns_density", 
    "second_person_pronouns_density", "third_person_pronouns_density", 
    "repetition_frequency",  
    'lexical_richness_MTLD', 'avg_dependency_score', 
    'avg_dependency_length', 'avg_discourse_markers', 'avg_modals', 'avg_epistemic_markers', 
    'avg_nominalisations', "function_word_density", "avg_entropy", "avg_caesura"
]
X = data[feature_columns]
y = data['generated']  # Replace 'label' with your actual label column

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

# Save the trained model
model_file = "saved_model.pkl"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

# Save the feature columns for prediction
features_file = "feature_columns.pkl"
joblib.dump(feature_columns, features_file)
print(f"Feature columns saved to {features_file}")
