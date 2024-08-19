import subprocess
import argparse
import os
import re
import time

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
    circuit_create_data_script = "circuit_create_data.py"
    generate_predictions_script = "generate.py"
    eval_main_script = "eval_main.py"
    # Construct paths
    cnf_input_file = os.path.join(data_dir, f"{circuit_name}.txt")
    source_file = os.path.join(data_dir, f"{circuit_name}_source.txt")
    target_file = os.path.join(data_dir, f"{circuit_name}_target.txt")
    prediction_output_file = os.path.join(data_dir, f"{circuit_name}_predictions.txt")
    result_file_cnfformer = os.path.join(data_dir, f"{circuit_name}_results.txt")
    
    # Step 3: Run circuit_create_data.py
    print(f"Running {circuit_create_data_script}...")
    run_script(circuit_create_data_script, '--input', cnf_input_file, '--output_source', source_file, '--output_target', target_file)    

    # Step 4: Run generate_predictions_dp.py
    print(f"Generating Predictions...")
    run_script(generate_predictions_script, '--input', source_file, '--model_path', model_path, '--output', prediction_output_file)
    
  # Step 5: Run cm_final_main.py
    print(f"Running eval_script...")
    output_cnfformer = run_script(eval_main_script, '--learned', prediction_output_file, '--target', target_file)
    precision_cnfformer, recall_cnfformer, f1_score_cnfformer = parse_results(output_cnfformer)
    print(f"CNFformer - Precision: {precision_cnfformer}, Recall: {recall_cnfformer}, F1 Score: {f1_score_cnfformer}")

    return  precision_cnfformer, recall_cnfformer, f1_score_cnfformer

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific circuit.')
    parser.add_argument('--circuit', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations to run the pipeline')
    args = parser.parse_args()
    precision_sum_cnfformer = 0.0
    recall_sum_cnfformer = 0.0
    f1_sum_cnfformer = 0.0
    
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}")
        precision_cf, recall_cf, f1_cf = run_pipeline(args.circuit,args.model_path,args.data_dir)
        
       
        precision_sum_cnfformer += precision_cf
        recall_sum_cnfformer += recall_cf
        f1_sum_cnfformer += f1_cf
    
   
    avg_precision_cnfformer = round(precision_sum_cnfformer / args.iterations, 2)
    avg_recall_cnfformer = round(recall_sum_cnfformer / args.iterations, 2)
    avg_f1_cnfformer = round(f1_sum_cnfformer / args.iterations, 2)
    print(f"Avg Precision: {avg_precision_cnfformer}\n Avg Recall: {avg_recall_cnfformer}\n Avg F1-Score: {avg_f1_cnfformer}\n")
    print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")