import json
import random
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Merge and shuffle JSON files.')
parser.add_argument('--data1', required=True, help='First input JSON file')
parser.add_argument('--data2', required=True, help='Second input JSON file')
parser.add_argument('--output', required=True, help='Output JSON file')

args = parser.parse_args()

# Read JSON data from two files
with open(args.data1, 'r', encoding='utf-8-sig') as file1:
    data1 = json.load(file1)

with open(args.data2, 'r', encoding='utf-8-sig') as file2:
    data2 = json.load(file2)

# Merge JSON data
merged_data = data1 + data2

print(f"len 1 {len(data1)}")
print(f"len 2 {len(data2)}")
print(f"new len {len(merged_data)}")

# Shuffle the merged data
random.shuffle(merged_data)

# Write the shuffled data to a new JSON file
with open(args.output, 'w', encoding='utf-8-sig') as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=2)
