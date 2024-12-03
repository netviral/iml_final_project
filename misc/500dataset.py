import csv
import random

def clean_essay_text(text):
    # Remove quotes at the start and end of the essay if they exist
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]  # Remove the first and last quote

    # Optionally: Remove any remaining quotes inside the text
    text = text.replace('"', '')  # Remove inner quotes (if you want)

    return text

def clean_and_select_rows(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)  # Read CSV into a dictionary
        fieldnames = reader.fieldnames  # Get original column headers

        # Separate rows into two categories: label 0 and label 1
        label_0_rows = []
        label_1_rows = []

        for row in reader:
            # Clean the essay text fields (assuming 'text' and 'generated' columns need cleaning)
            row['text'] = clean_essay_text(row['text'])

            # Check if the row belongs to label 0 or 1
            if row['generated'] == '0':
                label_0_rows.append(row)
            elif row['generated'] == '1':
                label_1_rows.append(row)

        # Select 500 rows for each label, if possible
        selected_label_0_rows = random.sample(label_0_rows, 500) if len(label_0_rows) >= 500 else label_0_rows
        selected_label_1_rows = random.sample(label_1_rows, 500) if len(label_1_rows) >= 500 else label_1_rows

        # Shuffle the selected rows from both categories
        selected_rows = selected_label_0_rows + selected_label_1_rows
        random.shuffle(selected_rows)

        # Write the shuffled rows to the new CSV file
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)

    print("CSV cleaning, selection, and shuffling completed. Saved to:", output_file)

# File paths
input_file = "../raw_data/Training_Essay_Data.csv"  # Your original file
output_file = "../processed_data/Final_Training_Dataset_Cleaned.csv"  # Output file with 1000 shuffled rows

# Run the cleaning and selection process
clean_and_select_rows(input_file, output_file)
