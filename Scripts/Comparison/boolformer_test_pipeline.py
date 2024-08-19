import numpy as np
import argparse
from boolformer import load_boolformer


def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert 'True' and 'False' strings to boolean values
            values = [val == 'True' for val in line.strip().split()]
            data.append(values)
    return data

# Function to read the first line from each file and append to lists
def read_first_line_to_lists(inputs_file, outputs_file):
    inputs_list = []
    outputs_list = []

    with open(inputs_file, 'r') as inputs_f, open(outputs_file, 'r') as outputs_f:
        while True:
            inputs_line = inputs_f.readline().strip()
            outputs_line = outputs_f.readline().strip()

            if not inputs_line or not outputs_line:
                break

            inputs_values = [val == 'True' for val in inputs_line.split()]
            outputs_value = outputs_line == 'True'

            inputs_list.append(inputs_values)
            outputs_list.append(outputs_value)

    return inputs_list, outputs_list

# Main function
def main(inputs_file, outputs_file,prediction):
    inputs_data = read_data(inputs_file)
    outputs_data = read_data(outputs_file)
    inputs = np.array(inputs_data)
    outputs = np.array(outputs_data).flatten()  # Flatten to ensure a 1D array for outputs
    # Load the Boolformer model
    boolformer_noisy = load_boolformer('noisy')
    pred_trees, errors, complexities = boolformer_noisy.fit([inputs], [outputs], verbose=False, beam_size=10, beam_type="search")
    
    for pred_tree in pred_trees:
        with open(prediction, 'w') as file:
            file.write(f"{pred_tree}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolformer Data Processing")
    parser.add_argument('--inputs', type=str, required=True, help="Path to the input file")
    parser.add_argument('--outputs', type=str, required=True, help="Path to the output file")
    parser.add_argument('--prediction', type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    main(args.inputs, args.outputs,args.prediction)
