import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize

# Load the dataset
df = pd.read_csv('new_small_data.csv')  # Replace with your actual file path

# Step 1: Cleaning the text
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Step 2: Function to compute average sentence length and variation
def compute_sentence_stats(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    total_words = len(words)
    total_sentences = len(sentences)
    average_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

    sentence_lengths = [len(sentence.split()) for sentence in sentences]

    mean_length = np.mean(sentence_lengths)
    std_deviation = np.std(sentence_lengths)
    
    return average_sentence_length, std_deviation

# Step 3: Process each row in the dataset
df['cleaned_text'] = df['text'].apply(clean_text)
df['average_sentence_length'], df['sentence_std_dev'] = zip(*df['cleaned_text'].apply(compute_sentence_stats))

# Step 4: Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot human-written (label 0) in red
plt.scatter(df[df['generated'] == 0]['average_sentence_length'], 
            df[df['generated'] == 0]['sentence_std_dev'], 
            color='red', label='Human Written', alpha=0.6)

# Plot AI-generated (label 1) in blue
plt.scatter(df[df['generated'] == 1]['average_sentence_length'], 
            df[df['generated'] == 1]['sentence_std_dev'], 
            color='blue', label='AI Generated', alpha=0.6)

# Add labels and title
plt.title('Variation in Sentence Lengths: Human vs AI')
plt.xlabel('Average Sentence Length')
plt.ylabel('Sentence Length Variation (Standard Deviation)')
plt.legend()

# Show plot
plt.grid(True)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

ax[0].scatter(df[df['generated'] == 0]['average_sentence_length'], 
              df[df['generated'] == 0]['sentence_std_dev'], 
              color='red', alpha=0.6)
ax[0].set_title('Human Written')
ax[0].set_xlabel('Average Sentence Length')
ax[0].set_ylabel('Sentence Length Variation')

ax[1].scatter(df[df['generated'] == 1]['average_sentence_length'], 
              df[df['generated'] == 1]['sentence_std_dev'], 
              color='blue', alpha=0.6)
ax[1].set_title('AI Generated')
ax[1].set_xlabel('Average Sentence Length')

plt.tight_layout()
plt.show()
