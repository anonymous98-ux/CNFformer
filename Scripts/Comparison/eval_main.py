from f1_check import batch_equisatisfiability_check
import argparse
def read_cnf_formulas_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def main(predicted_file, ground_truth_file, alpha=2):
    predicted_cnfs = read_cnf_formulas_from_file(predicted_file)
    ground_truth_cnfs = read_cnf_formulas_from_file(ground_truth_file)

    # Assuming all predictions are valid for simplification
    valid_predictions_count = len(predicted_cnfs)

    precision, recall, f1_score = batch_equisatisfiability_check(predicted_cnfs, ground_truth_cnfs, valid_predictions_count, alpha)
    print(f"Precision: {round(precision,2)}")
    print(f"Recall: {round(recall,2)}")
    print(f"f1_score: {round(f1_score,2)}")
    return precision, recall, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CNF formula predictions')
    parser.add_argument('--learned', type=str, required=True, help='Path to the source data file')
    parser.add_argument('--target', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()


    main(args.learned, args.target)