import spacy
import pandas as pd
import re
import numpy as np
from lexicalrichness import LexicalRichness
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords
from collections import Counter
import numpy as np
import os
import csv
import ssl
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
# Load spaCy model globally
nlp = spacy.load("en_core_web_sm")

# SSL workaround for downloading NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('cmudict')
# nltk.download('stopwords')
# nltk.download('wordnet')


# Function to calculate sentence complexity based on dependency labels
def sent_complexity_structure(doc):
    dep_labels = {"acl", "conj", "advcl", "ccomp", "csubj", "discourse", "parataxis"}
    return sum(1 for token in doc if token.dep_ in dep_labels)

# Function to calculate dependency score
def calculate_dep_score(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    return np.mean([sent_complexity_structure(sent) for sent in sentences])

# Function to calculate dependency depth
def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def calculate_dep_length(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    return np.mean([walk_tree(sent.root, 0) for sent in sentences])

# Function to calculate MTLD (Lexical Richness)
def calculate_lex_richness_MTLD2(text):
    lex = LexicalRichness(text)
    return lex.mtld()

# Function to count discourse markers
discourse = pd.read_csv("markers/connectives_discourse_markers_PDTB.txt", sep="\'", encoding="UTF-8", header=None, usecols=[1,3])
discourse[3] = discourse[3].apply(lambda x: x.replace("t_conn_", ""))
discourse[1] = discourse[1].apply(lambda x: " " + x + " ")

def count_discourse_markers(text):
    count = 0
    for marker in discourse[1]:
        count += text.count(marker)
    return count

# Function to count modals
modals = pd.read_csv("markers/modals.csv", sep=",", encoding="UTF-8", header=None)
modals[0] = modals[0].apply(lambda x: x.replace('_', ' '))

def count_total_modals(text):
    count = 0
    for modal in modals[0]:
        count += text.count(modal)
    return count

# Function to count epistemic markers
def find_epistemic_markers(text):
    ep_markers = []
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)(?:\s\w+)?(?:\s\w+)?\s(?:believes?|thinks?|means?|worry|worries|know|guesse?s?|assumes?)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:It|it)\sis\s(?:believed|known|assumed|thought)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:I|We|we)\s(?:am|are)\s(?:thinking|guessing)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)(?:\s\w+)?\s(?:do|does)\snot\s(?:believe?|think|know)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)\swould(?:\s\w+)?(?:\snot)?\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"I\sam\s(?:afraid|sure|confident)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:My|my|Our|our)\s(?:experience|opinion|belief|knowledge|worry|worries|concerns?|guesse?s?)\s(?:is|are)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"[In]n\s(?:my|our)(?:\s\w+)?\sopinion", text))
    ep_markers.extend(re.findall(r"As\sfar\sas\s(?:I|We|we)\s(?:am|are)\sconcerned", text))
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)\s(?:can|could|may|might)(?:\s\w+)?\sconclude\s(?:that)?", text))
    ep_markers.extend(re.findall(r"I\s(?:am\swilling\sto|must)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"One\s(?:can|could|may|might)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"[Oo]ne\s(?:can|could|may|might)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"[Ii]t\sis\s(?:obvious|(?:un)?clear)", text))
    ep_markers.extend(re.findall(r"[Ii]t\s(?:seems|feels|looks)", text))
    return len(ep_markers)

# Function to count nominalisations
def nominalisation_counter(text):
    suffixes_n = r'\b[A-Z]*\w+(?:tion|ment|ance|ence|ion|it(?:y|ies)|ness|ship)(?:s|es)?\b'
    nom_nouns = [token.text for token in nlp(text) if token.pos_ == 'NOUN' and re.match(suffixes_n, token.text)]
    return len(nom_nouns)

# Function to calculate the average number of features per sentence
def average_per_sentence(feature, sentences_count):
    if sentences_count == 0:
        return 0
    return feature / sentences_count

d = cmudict.dict()

# Function Definitions
def count_syllables(word):
    """Returns the number of syllables in a word."""
    try:
        return max([len([phoneme for phoneme in pronunciation if phoneme[-1].isdigit()]) for pronunciation in d[word.lower()]])
    except KeyError:
        return 1  # Default to 1 syllable for unknown words

def is_complex_word(word):
    """Determines if a word is complex (3 or more syllables)."""
    return count_syllables(word) >= 3
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    return " ".join(lemmatized_words)

def calculate_cosine_similarity(sentences):
    """Calculates redundancy and coherence using cosine similarity with lemmatization."""
    
    # Lemmatize all sentences
    lemmatized_sentences = [lemmatize_sentence(sentence) for sentence in sentences]
    
    # Convert lemmatized sentences to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lemmatized_sentences).toarray()
    
    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Redundancy: All unique pairs
    n = len(sentences)
    redundancy_scores = [similarity_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    redundancy = np.mean(redundancy_scores) if redundancy_scores else 0
    
    # Coherence: Consecutive pairs
    coherence_scores = [similarity_matrix[i][i + 1] for i in range(n - 1)]
    coherence = np.mean(coherence_scores) if coherence_scores else 0
    
    return {
        "Redundancy": redundancy,
        "Coherence": coherence
    }

def calculate_stopword_ratio(words):
    """Calculates the ratio of stopwords to total words."""
    stop_words = set(stopwords.words('english'))
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    return stopword_count / len(words) if words else 0

def calculate_rare_word_ratio(words, threshold=0.01):
    """Calculates the ratio of rare words to total words."""
    word_frequencies = Counter(words)
    rare_words = [word for word, freq in word_frequencies.items() if freq / len(words) < threshold]
    return len(rare_words) / len(words) if words else 0

def calculate_zipfs_law_adherence(words):
    """Calculates Zipf's Law adherence using rank-frequency distribution."""
    word_frequencies = Counter(words)
    sorted_frequencies = sorted(word_frequencies.values(), reverse=True)
    rank_frequency = [(rank + 1) * freq for rank, freq in enumerate(sorted_frequencies)]
    deviation = np.std(rank_frequency) / np.mean(rank_frequency) if rank_frequency else 0
    return 1 - deviation  # Higher adherence to Zipf's law if deviation is small


# Load discourse markers from PDTB list
discourse = pd.read_csv("markers/connectives_discourse_markers_PDTB.txt", sep="\'", encoding="UTF-8", header=None, usecols=[1,3])
discourse[3] = discourse[3].apply(lambda x: x.replace("t_conn_", ""))
discourse[1] = discourse[1].apply(lambda x: " " + x + " ")

# Counts the total number of discourse markers in the text
def count_discourse_markers(text):
    count = 0
    for marker in discourse[1]:
        count += text.count(marker)  # Count occurrences of each marker
    return count
# Load modals list
modals = pd.read_csv("markers/modals.csv", sep=",", encoding="UTF-8", header=None)
modals[0] = modals[0].apply(lambda x: x.replace('_', ' '))
# Counts the total number of modals in the text
def count_total_modals(text):
    count = 0
    for modal in modals[0]:
        count += text.count(modal)  # Count occurrences of each modal
    return count


def calculate_sentence_complexity(sentences):
    """
    Calculates the complexity as the ratio of clauses to sentences,
    while skipping sentences identified as lists.
    """
    clause_count = 0
    valid_sentence_count = 0

    for sentence in sentences:
        # Skip sentences identified as lists
        if re.match(r"^\d+\.|\*|\-|\([a-zA-Z]*\)|[a-zA-Z]\)", sentence.strip()):
            continue
        
        # Otherwise, count clauses (assume each ',' introduces a clause)
        clause_count += sentence.count(',') + 1
        valid_sentence_count += 1

    return clause_count / valid_sentence_count if valid_sentence_count else 0


def clean_text(text):
    """Cleans and preprocesses the input text."""
    return text
    # return re.sub(r'\s+', ' ', text.strip())

def tokenize_words(text):
    """Tokenizes the text and filters out punctuation."""
    words = word_tokenize(text)
    return [word for word in words if word.isalpha()]

def calculate_variation(values):
    """
    Calculates the variation (standard deviation) of a list of values.
    """
    return np.std(values)

def calculate_sentence_complexity(sentences):
    """
    Calculates complexity and variation of clauses per sentence.
    """
    clause_counts = []
    valid_sentences = []

    for sentence in sentences:
        # Skip sentences identified as lists
        if re.match(r"^\d+\.|\*|\-|\([a-zA-Z]*\)|[a-zA-Z]\)", sentence.strip()):
            continue
        
        # Count clauses (assume each ',' introduces a clause)
        clauses = sentence.count(',') + 1
        clause_counts.append(clauses)
        valid_sentences.append(sentence)

    # Calculate average complexity and variation
    complexity_variation = calculate_variation(clause_counts) if clause_counts else 0

    return complexity_variation

def analyze_sentence_lengths(sentences):
    """
    Analyzes sentence length and calculates variation.
    """
    lengths = [len(sentence.split()) for sentence in sentences]
    avg_length = np.mean(lengths) if lengths else 0
    length_variation = calculate_variation(lengths) if lengths else 0

    return avg_length, length_variation

def pos_diversity_mtld(text, window_size=10, threshold=0.1):
    """
    Calculate the MTLD-like POS diversity score for a given text.
    Args:
        text (str): The input text to analyze.
        window_size (int): The size of the sliding window in tokens (default 10).
        threshold (float): The threshold for diversity difference to reset the count (default 0.1).
    Returns:
        float: The MTLD-like POS diversity score.
    """
    # Tokenize the text and calculate POS tags
    tokens = word_tokenize(text)
    tags = [tag for word, tag in pos_tag(tokens)]
    
    # Function to calculate POS diversity in a given window
    def calculate_diversity(window):
        return len(set(window))  # Count unique POS tags in the window

    pos_diversities = []
    for i in range(0, len(tags) - window_size + 1):
        window = tags[i:i + window_size]
        diversity = calculate_diversity(window)
        pos_diversities.append(diversity)

    # Calculate MTLD-like score
    n = len(pos_diversities)
    if n == 0:
        return 0

    mtld_score = 0
    current_sequence_length = 1

    for i in range(1, n):
        # If the difference in diversity exceeds the threshold, reset the sequence
        if abs(pos_diversities[i] - pos_diversities[i - 1]) > threshold:
            mtld_score += current_sequence_length
            current_sequence_length = 1  # reset sequence length
        else:
            current_sequence_length += 1

    mtld_score += current_sequence_length  # add the last sequence
    return mtld_score / n  # normalize by number of windows


def noun_verb_ratio(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    return noun_count / verb_count if verb_count != 0 else 0

def adj_adv_ratio(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    return (adj_count + adv_count) / len(tokens) if len(tokens) > 0 else 0

def grammar_error_density(text):
    blob = TextBlob(text)
    return len(blob.correct().sentences) - len(blob.sentences)

def passive_active_ratio(text):
    sentences = nltk.sent_tokenize(text)
    passive_count = sum(1 for sent in sentences if "by" in sent)
    active_count = len(sentences) - passive_count
    return passive_count / active_count if active_count != 0 else 0

def punctuation_frequency(text):
    punct_counts = Counter(char for char in text if char in string.punctuation)
    return punct_counts

def pronoun_usage(text, total_words):
    tokens = word_tokenize(text.lower())
    pronouns = {
        "first_person": ["i", "me", "we", "us"],
        "second_person": ["you", "your"],
        "third_person": ["he", "she", "it", "they", "them"]
    }
    
    usage = {
        "first_person": sum(tokens.count(p) for p in pronouns["first_person"]),
        "second_person": sum(tokens.count(p) for p in pronouns["second_person"]),
        "third_person": sum(tokens.count(p) for p in pronouns["third_person"])
    }
    
    # Calculate density (pronoun count / total words)
    pronoun_density = {
        "first_person_density": usage["first_person"] / total_words,
        "second_person_density": usage["second_person"] / total_words,
        "third_person_density": usage["third_person"] / total_words
    }
    
    return pronoun_density

def repetition_frequency(text):
    words = word_tokenize(text.lower())
    word_count = len(words)
    repeated_words = sum(1 for _, count in Counter(words).items() if count > 1)
    return repeated_words / word_count if word_count > 0 else 0

def calculate_entropy(text):
    char_counts = Counter(text)
    total_chars = len(text)
    probabilities = [count / total_chars for count in char_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def count_function_words(text, function_words):
    """
    Counts the number of function words in the given text.
    """
    words = text.lower().split()  # Tokenize by splitting on whitespace
    count = sum(1 for word in words if word in function_words)
    return count

def load_function_words(file_path):
    """
    Reads the function words from a file and returns them as a set.
    Each function word is assumed to be on a separate line.
    """
    with open(file_path, 'r') as file:
        function_words = {line.strip().lower() for line in file}
    return function_words
def function_word_density(text, function_words):
    """
    Calculates the proportion of function words in the text.
    """
    words = text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return 0
    function_word_count = count_function_words(text, function_words)
    return function_word_count / total_words

def caesura_count(text):
    # Define caesura punctuation marks
    caesura_marks = [',', ';', '—', '-', '...', '(', ')']
    count = sum(text.count(mark) for mark in caesura_marks)
    return count

def average_caesura_count(text):
    # Split text into sentences
    sentences = text.split('.')
    total_caesura = caesura_count(text)
    sentence_count = len(sentences)
    return total_caesura / sentence_count if sentence_count > 0 else 0

def process_text(text):
    # Text Preprocessing
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    words = tokenize_words(cleaned_text)
    paragraphs = cleaned_text.split('\n\n')

    # Basic Counts
    total_words = len(words)
    unique_words = len(set(words))
    total_characters = sum(len(word) for word in words)
    total_sentences = len(sentences)
    total_paragraphs = len(paragraphs)

    # Metrics Calculation
    average_word_length = np.mean([len(word) for word in words])
    total_syllables = sum(count_syllables(word) for word in words)
    complex_word_count = sum(is_complex_word(word) for word in words)
    stopword_ratio = calculate_stopword_ratio(words)
    rare_word_ratio = calculate_rare_word_ratio(words)
    lexical_diversity = calculate_lex_richness_MTLD2(text)
    sentence_complexity = calculate_sentence_complexity(sentences)
    # Define the file path
    file_path = "markers/function-words-short.txt"

    # Load the function words
    function_words = load_function_words(file_path)
    # Analyze the text
    total_function_words = count_function_words(text, function_words)
    fwd = function_word_density(text, function_words)
    avg_entropy=calculate_entropy(text)
    avg_caesura = average_caesura_count(text)

    # Readability Scores
    flesch_reading_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    gunning_fog_index = 0.4 * ((total_words / total_sentences) + (complex_word_count / total_words) * 100)

    # Cosine Similarity Metrics
    cosine_scores = calculate_cosine_similarity(sentences)

    # Filter out non-list sentences for length analysis
    non_list_sentences = [s for s in sentences if not re.match(r"^\d+\.|\*|\-|\([a-zA-Z]*\)|[a-zA-Z]\)", s.strip())]

    # Calculate sentence complexity and variation
    complexity_variation = calculate_sentence_complexity(non_list_sentences)

    # Calculate sentence length and variation
    avg_length, length_variation = analyze_sentence_lengths(non_list_sentences)
    
    zipfs_law_adherence = calculate_zipfs_law_adherence(text)
    pos_diversity_value = pos_diversity_mtld(text)
    noun_verb_ratio_value = noun_verb_ratio(text)
    adjective_adverb_ratio_value = adj_adv_ratio(text)
    grammar_error_density_value = grammar_error_density(text)
    passive_to_active_ratio_value = passive_active_ratio(text)
    punctuation_freq = punctuation_frequency(text)
    total_punctuations = sum(punctuation_freq.values())  # Get total punctuation count
    punctuation_density = total_punctuations / total_words
    pronoun_density = pronoun_usage(text, total_words)
    repetition_frequency_value = repetition_frequency(text)
    doc = nlp(text)
    # Get various features
    dep_score = calculate_dep_score(text)
    dep_length = calculate_dep_length(text)
    lex_richness = calculate_lex_richness_MTLD2(text)
    discourse_count = count_discourse_markers(text)
    modals_count = count_total_modals(text)
    epistemic_count = find_epistemic_markers(text)
    nominalisations_count = nominalisation_counter(doc)
    # Average counts per sentence (you can apply this for any feature you want)
    avg_dep_score = average_per_sentence(dep_score, total_sentences)
    avg_dep_length = average_per_sentence(dep_length, total_sentences)
    avg_discourse = average_per_sentence(discourse_count, total_sentences)
    avg_modals = average_per_sentence(modals_count, total_sentences)
    avg_epistemic = average_per_sentence(epistemic_count, total_sentences)
    avg_nominalisations = average_per_sentence(nominalisations_count, total_sentences)

    return {
            "total_words": total_words,
            "unique_words": unique_words,
            "total_characters": total_characters,
            "total_sentences": total_sentences,
            "total_paragraphs": total_paragraphs,
            "average_word_length": average_word_length,
            "complex_word_density": complex_word_count / total_words,
            "stopword_ratio": stopword_ratio,
            "rare_word_ratio": rare_word_ratio,
            "sentence_complexity": sentence_complexity,
            "flesch_reading_ease": flesch_reading_ease,
            "gunning_fog_index": gunning_fog_index,
            "cosine_similarity_redundancy": cosine_scores["Redundancy"],
            "cosine_similarity_coherence": cosine_scores["Coherence"],
            "sentence_length_variation": length_variation,
            "complexity_variation": complexity_variation,
            "avg_sentence_length": avg_length,
            "zipfs_law_adherence": zipfs_law_adherence,
            "pos_diversity": pos_diversity_value,
            "noun_verb_ratio": noun_verb_ratio_value,
            "adjective_adverb_ratio": adjective_adverb_ratio_value,
            "grammar_error_density": grammar_error_density_value,
            "passive_to_active_ratio": passive_to_active_ratio_value,
            "punctuation_density": punctuation_density,
            "first_person_pronouns_density": pronoun_density['first_person_density'],
            "second_person_pronouns_density": pronoun_density['second_person_density'],
            "third_person_pronouns_density": pronoun_density['third_person_density'],
            "repetition_frequency": repetition_frequency_value,
            'dependency_score': dep_score,
            'dependency_length': dep_length,
            'lexical_richness_MTLD': lex_richness,
            'discourse_markers_count': discourse_count,
            'modals_count': modals_count,
            'epistemic_markers_count': epistemic_count,
            'nominalisations_count': nominalisations_count,
            'avg_dependency_score': avg_dep_score,
            'avg_dependency_length': avg_dep_length,
            'avg_discourse_markers': avg_discourse,
            'avg_modals': avg_modals,
            'avg_epistemic_markers': avg_epistemic,
            'avg_nominalisations': avg_nominalisations,
            "function_word_density":fwd,
            "avg_entropy":avg_entropy,
            "avg_caesura": avg_caesura
        }
