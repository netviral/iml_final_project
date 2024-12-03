
import os
import csv
import pandas as pd
from helper_functions import process_text

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
input_file = "./processed_data/Final_Training_Dataset_Cleaned.csv"
output_file = "./processed_data/Final_Training_Done.csv"
log_file = "progress.log"
process_csv(input_file,output_file,log_file)