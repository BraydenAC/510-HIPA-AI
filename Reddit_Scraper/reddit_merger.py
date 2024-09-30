import os
import csv

# Directory containing the individual CSV files
input_dir = 'reddit_posts'
output_file = 'merged_reddit_posts.csv'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Boolean if output file already exists
output_exists = os.path.exists(output_file)

# Open the output file for writing
with open(output_file, 'a' if output_exists else 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)

    if not output_exists:
        writer.writerow(['title', 'selftext', 'created_utc'])  # Write the header

    # Iterate through each CSV file and write its contents to the output file
    for csv_file in csv_files:
        with open(os.path.join(input_dir, csv_file), 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header row
            for row in reader:
                writer.writerow(row)

        # Delete the individual CSV file after its contents have been appended
        os.remove(os.path.join(input_dir, csv_file))

print(f"All posts have been merged into '{output_file}' and individual files have been deleted.")
