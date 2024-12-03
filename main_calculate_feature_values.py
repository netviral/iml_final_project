
import os
import csv
import pandas as pd
from helper_functions import process_text

def process_csv(input_file, output_file, log_file):
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
    start_row = 0
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            start_row = int(f.read().strip())

    for chunk in pd.read_csv(input_file, chunksize=1, skiprows=range(1, start_row + 1), encoding='utf-8', iterator=True):
        row = chunk.iloc[0]
        text = row["text"]
        generated = row["generated"]
        processed_data = process_text(text)
        output_row = [text, generated] + list(processed_data.values())
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_row)
        start_row += 1
        with open(log_file, 'w') as f:
            f.write(str(start_row))

        print(f"Processed row {start_row}")

input_file = "./processed_data/Final_Training_Dataset_Cleaned.csv"
output_file = "./processed_data/Final_Training_Done.csv"
log_file = "progress.log"
process_csv(input_file,output_file,log_file)