import pandas as pd
import json
import random

# Function to load the entire JSON file into a Pandas DataFrame
def load_json_to_df(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.json_normalize(data, 'items', ['user_id', 'items_count', 'steam_id', 'user_url'])

# Function to load the review JSON file into a Pandas DataFrame
def load_review_json_to_df(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.json_normalize(data, 'reviews', ['user_id'])



def read_first_n_records(file_path, n=5):
    records = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line.strip()))
    return records

# Function to read the first few raw lines from the file
def read_first_n_raw_lines(file_path, n=5):
    lines = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.strip())
    return lines



    """
    Reads a large newline-delimited JSON file and writes a smaller sampled JSON file.
    
    Parameters:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output JSON file.
        sample_size (int): Number of records to sample.
        
    Returns:
        None
    """
    
    # Count total number of records in the input file
    total_records = 0
    with open(input_file_path, 'r') as infile:
        for line in infile:
            total_records += 1
    
    # Generate random line numbers to sample
    sample_line_nums = random.sample(range(total_records), sample_size)
    sample_line_nums.sort()
    
    # Read input file and write sampled records to output file
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        current_line_num = 0
        sampled_count = 0
        for line in infile:
            if current_line_num == sample_line_nums[sampled_count]:
                outfile.write(line)
                sampled_count += 1
                if sampled_count >= sample_size:
                    break
            current_line_num += 1

import ast  # For safely evaluating a string containing a Python literal or container display

def sample_python_dict_file(input_file_path, output_file_path, sample_size):
    """
    Reads a large file with Python-style dictionaries and writes a smaller sampled JSON file.
    
    Parameters:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output JSON file.
        sample_size (int): Number of records to sample.
        
    Returns:
        None
    """
    
    # Count total number of records in the input file
    total_records = 0
    with open(input_file_path, 'r') as infile:
        for line in infile:
            total_records += 1
    
    # Generate random line numbers to sample
    sample_line_nums = random.sample(range(total_records), sample_size)
    sample_line_nums.sort()
    
    # Read input file and write sampled records to output file
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        current_line_num = 0
        sampled_count = 0
        for line in infile:
            if current_line_num == sample_line_nums[sampled_count]:
                # Safely evaluate the Python dictionary string to a dictionary
                py_dict = ast.literal_eval(line.strip())
                # Convert the Python dictionary to a JSON object
                json_str = json.dumps(py_dict)
                outfile.write(json_str + "\n")
                sampled_count += 1
                if sampled_count >= sample_size:
                    break
            current_line_num += 1

def convert_to_well_formed_json(input_file_path, output_file_path):
    """
    Reads a file with Python-style dictionaries and writes a well-formed JSON file.
    
    Parameters:
        input_file_path (str): Path to the input file containing Python-style dictionaries.
        output_file_path (str): Path to the output JSON file.
        
    Returns:
        None
    """
    
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Safely evaluate the Python dictionary string to a dictionary
            py_dict = ast.literal_eval(line.strip())
            # Convert the Python dictionary to a JSON object
            json_str = json.dumps(py_dict)
            outfile.write(json_str + "\n")