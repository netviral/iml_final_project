import csv
import re

def clean_essay_text(text):
    # Remove quotes at the start and end of the essay if they exist
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]  # Remove the first and last quote

    # Optionally: Remove any remaining quotes inside the text
    text = text.replace('"', '')  # Remove inner quotes (if you want)

    return text

def clean_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)  # Read CSV into a dictionary
        fieldnames = reader.fieldnames  # Get original column headers

        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                # Clean the essay text fields (assuming 'text' and 'generated' columns need cleaning)
                row['text'] = clean_essay_text(row['text'])
                row['generated'] = clean_essay_text(row['generated'])

                # Write cleaned row to the new CSV file
                writer.writerow(row)

    print("CSV cleaning completed and saved to:", output_file)

# File paths
input_file = "training_human.csv"
output_file = "cleaned_training_human.csv"

# Run the cleaning process
clean_csv(input_file, output_file)
