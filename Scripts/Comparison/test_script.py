import subprocess
import argparse
import os
import re
import time
import numpy as np

def run_script(script_name, *args):
    """Utility function to run a script with the given arguments."""
    result = subprocess.run(['python', script_name] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
        raise Exception(f"Failed to run {script_name}")
    print(result.stdout)
    return result.stdout

def parse_results(output):
    """Parse the output from cm_final_main.py to extract precision, recall, and f1_score."""
    precision = float(re.search(r'Precision: (\d+\.\d+)', output).group(1))
    recall = float(re.search(r'Recall: (\d+\.\d+)', output).group(1))
    f1_score = float(re.search(r'f1_score: (\d+\.\d+)', output).group(1))
    return precision, recall, f1_score

def run_pipeline(circuit_name,model_path,data_dir):

    circuit_create_data_script = "gen_data.py"
    boolformer_convert_script = "boolformer_convert_pipeline.py"
    boolformer_test_script = "boolformer_test_pipeline.py"
    transform_tocnf_script = "transform_tocnf.py"
    generate_predictions_script = "generate.py"
    eval_main_script = "eval_main.py"
    
    # Construct paths
    cnf_input_file = os.path.join(data_dir, f"{circuit_name}.txt")
    source_file = os.path.join(data_dir, f"{circuit_name}_cf_source.txt")
    source_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_source.txt")
    target_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_target.txt")
    target_file = os.path.join(data_dir, f"{circuit_name}_cf_target.txt")
    inputs_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_inputs.txt")
    outputs_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_outputs.txt")
    prediction_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_prediction.txt")
    cnf_file_boolformer = os.path.join(data_dir, f"{circuit_name}_bool_converted_tocnf.txt")
    prediction_output_file = os.path.join(data_dir, f"{circuit_name}_cf_predictions.txt")


    run_script(circuit_create_data_script, '--input', cnf_input_file, '--output_source', source_file, '--output_target', target_file)       
    run_script(boolformer_convert_script, '--input',source_file, '--output_inputs', inputs_file_boolformer, '--output_outputs', outputs_file_boolformer)
    run_script(boolformer_test_script, '--inputs', inputs_file_boolformer, '--outputs', outputs_file_boolformer, '--prediction', prediction_file_boolformer)
    run_script(transform_tocnf_script, '--input', prediction_file_boolformer, '--output', cnf_file_boolformer)

    output_boolformer = run_script(eval_main_script, '--learned', cnf_file_boolformer, '--target', cnf_input_file)
    precision_boolformer, recall_boolformer, f1_score_boolformer = parse_results(output_boolformer)
    print(f"Boolformer - Precision: {precision_boolformer}, Recall: {recall_boolformer}, F1 Score: {f1_score_boolformer}")
    run_script(generate_predictions_script, '--input', source_file, '--model_path', model_path, '--output', prediction_output_file)
    
    output_cnfformer = run_script(eval_main_script, '--learned', prediction_output_file, '--target', target_file)
    precision_cnfformer, recall_cnfformer, f1_score_cnfformer = parse_results(output_cnfformer)
    print(f"CNFformer - Precision: {precision_cnfformer}, Recall: {recall_cnfformer}, F1 Score: {f1_score_cnfformer}")

    return precision_boolformer, recall_boolformer, f1_score_boolformer, precision_cnfformer, recall_cnfformer, f1_score_cnfformer

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific circuit.')
    parser.add_argument('--circuit', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    args = parser.parse_args()
    precision_sum_boolformer = 0.0
    recall_sum_boolformer = 0.0
    f1_sum_boolformer = 0.0
    precision_sum_cnfformer = 0.0
    recall_sum_cnfformer = 0.0
    f1_sum_cnfformer = 0.0
    
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}")
        precision_bf, recall_bf, f1_bf, precision_cf, recall_cf, f1_cf = run_pipeline(args.circuit,args.model_path,args.data_dir)
        
        precision_sum_boolformer += precision_bf
        recall_sum_boolformer += recall_bf
        f1_sum_boolformer += f1_bf
        precision_sum_cnfformer += precision_cf
        recall_sum_cnfformer += recall_cf
        f1_sum_cnfformer += f1_cf
    
    avg_precision_boolformer = round(precision_sum_boolformer / args.iterations, 2)
    avg_recall_boolformer = round(recall_sum_boolformer / args.iterations, 2)
    avg_f1_boolformer = round(f1_sum_boolformer / args.iterations, 2)
    avg_precision_cnfformer = round(precision_sum_cnfformer / args.iterations, 2)
    avg_recall_cnfformer = round(recall_sum_cnfformer / args.iterations, 2)
    avg_f1_cnfformer = round(f1_sum_cnfformer / args.iterations, 2)

    # Print average results
    print(f"\nAverage results after {args.iterations} iterations:")
    print(f"Boolformer - Precision: {avg_precision_boolformer}, Recall: {avg_recall_boolformer}, F1 Score: {avg_f1_boolformer}")
    print(f"CNFformer - Precision: {avg_precision_cnfformer}, Recall: {avg_recall_cnfformer}, F1 Score: {avg_f1_cnfformer}")
    print(f"BOOLFORMER - Avg Precision: {avg_precision_boolformer}\n Avg Recall: {avg_recall_boolformer}\n Avg F1-Score: {avg_f1_boolformer}\n")
    print(f"CNFFORMER - Avg Precision: {avg_precision_cnfformer}\n Avg Recall: {avg_recall_cnfformer}\n Avg F1-Score: {avg_f1_cnfformer}\n")
    print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")