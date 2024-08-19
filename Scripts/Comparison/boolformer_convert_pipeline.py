import numpy as np
import argparse
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()
    return data
def write_data(file_path, data):
    np.savetxt(file_path, data, fmt='%s')

def main(input_file,output_inputs,output_outputs):
    data = read_data(input_file)
    assignments = data.split(',')

    inputs = []
    outputs = []

    for assignment in assignments:
        parts = assignment.split()
        current_input = []
        for i in range(0, len(parts), 2):
            key = parts[i]
            value = parts[i + 1]
            if 'x' in key:
                current_input.append(True if value == '1' else False)
            elif 'boolean' in key:
                outputs.append(True if value == '1' else False)
        inputs.append(current_input)

    # Convert to numpy arrays
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # Shuffle the data
    shuffled_indices = np.arange(inputs.shape[0])
    np.random.shuffle(shuffled_indices)
    inputs = inputs[shuffled_indices]
    outputs = outputs[shuffled_indices]
    # Write the processed arrays to output files
    write_data(output_inputs, inputs)
    write_data(output_outputs, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNF Formula Generator")
    parser.add_argument('--input', type=str, required=True, help="Number of variables in the CNF formulas")
    parser.add_argument('--output_inputs', type=str, required=True, help="Number of clauses in the CNF formulas")
    parser.add_argument('--output_outputs', type=str, required=True, help="Number of clauses in the CNF formulas")
    args = parser.parse_args()
    main(args.input,args.output_inputs,args.output_outputs)