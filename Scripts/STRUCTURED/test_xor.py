import logging
import sys
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import random
import torch
import subprocess
import time
import pickle
import concurrent.futures
from gen_cnf_formula import generate_cnf_formula
import cProfile
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from transformers import LEDTokenizer, LEDForConditionalGeneration
from torch.utils.data import DataLoader
from data_utils_gen_pred import CNFDataset, collate_fn
from f1_check import batch_equisatisfiability_check


def generate_predictions(internal_nodes, source_file_name, target_file_name, model_path):
    source_file,target_file = source_file_name,target_file_name
    def load_source_data(source_file):
        with open(source_file, 'r') as src_file:
            return [line.strip() for line in src_file.readlines()]
        
    print('Script Started')
    output_path = f"{internal_nodes}internal_nodes_predictions.txt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    # Load source data
    source_data = load_source_data(source_file)
    print('Test data loaded')
    # Initialize tokenizer and model
    tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
    model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
    state_dict = torch.load(model_path, map_location = device)['model_state_dict']
    adjusted_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    # Load the adjusted state dict
    model.load_state_dict(adjusted_state_dict)
    
    
  #  device = 'cpu'
    model.to(device)
    model.eval()

    # Create data loader for the source data
    test_dataset = CNFDataset(source_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    print('Created data loader')

    # Predict and format CNF expressions
    predictions = []
    print('Generating predictions')
    with torch.no_grad():
        for batch in test_dataloader:
            # Move input tensors to the same device as the model
            src_ids, attention_masks = batch[0].to(device), batch[1].to(device)

            # Generate predictions
            outputs = model.generate(input_ids=src_ids, attention_mask=attention_masks, max_length=1000)
            decoded_predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            predictions.extend(decoded_predictions)

    # Save predictions to a file
    with open(output_path, "w") as file:
        for pred in predictions:
            file.write(f"{pred}\n")
    return output_path,target_file

def main(internal_nodes,source_file_name, target_file_name, model_path):
    predicted_file,ground_truth_file = generate_predictions(internal_nodes,source_file_name, target_file_name, model_path)
    def read_cnf_formulas_from_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

   
    predicted_cnfs = read_cnf_formulas_from_file(predicted_file)
    ground_truth_cnfs = read_cnf_formulas_from_file(ground_truth_file)

    valid_predictions_count = len(predicted_cnfs)

    precision, recall, f1_score = batch_equisatisfiability_check(predicted_cnfs, ground_truth_cnfs, valid_predictions_count,alpha = 0.5)
   
    print(f"Precision: {round(precision,2)}")
    print(f"Recall: {round(recall,2)}")
    print(f"f1_score: {round(f1_score,2)}")
    print(f"PROCESS COMPLETED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNF Formula Generator")
    parser.add_argument('--model_path', type=str, required=True, help="Directory to save source and target files")
    parser.add_argument('--internal_nodes', type=int, required=True, help="Number of variables in the CNF formulas")
    parser.add_argument('--source_file', type=str, required=True, help="Number of clauses in the CNF formulas")
    parser.add_argument('--target_file', type=str, required=True, help="Number of clauses in the CNF formulas")
    
    args = parser.parse_args()

    main(args.internal_nodes, args.source_file,args.target_file, args.model_path)


