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
from cm_final_adjusted import batch_equisatisfiability_check


def walkSAT(clauses, p=0.5, max_flips=1000):
    
    # Helper function to compute break count for a variable
    start_time = time.time()
    def compute_break(variable, assignment, pos_lit_clause, neg_lit_clause):
      break_count = 0
      # Adjust the indexing by subtracting 1 from 'variable'
      clauses_to_check = pos_lit_clause[variable] if assignment[variable - 1] else neg_lit_clause[variable]
      for clause_index in clauses_to_check:
          if true_literal_count(clauses[clause_index], assignment) == 1:
              break_count += 1
      return break_count


    # Helper function to count true literals in a clause
    def true_literal_count(clause, assignment):
        return sum(assignment[abs(lit) - 1] == (lit > 0) for lit in clause)


    # Random initial assignment
    n_vars = max(abs(lit) for clause in clauses for lit in clause)

    # Random initial assignment (index 0 is unused)

    assignment = [random.choice([False, True]) for _ in range(n_vars)]

    # Initialize positive and negative literal clauses
    pos_lit_clause = [[] for _ in range(n_vars + 1)]
    neg_lit_clause = [[] for _ in range(n_vars + 1)]

    # Populate positive and negative literal clauses
    for i, clause in enumerate(clauses):
        for lit in clause:
            if lit > 0:
                pos_lit_clause[lit].append(i)
            else:
                neg_lit_clause[abs(lit)].append(i)


   
    for _ in range(max_flips):
        # Find unsatisfied clauses
        unsatisfied = [clause for clause in clauses if not true_literal_count(clause, assignment)]
        if not unsatisfied:
            return assignment  # SATISFIABLE

        # Choose a random unsatisfied clause
        clause = random.choice(unsatisfied)
        min_break = float('inf')
        candidates = []
        # Find the variable to flip
        for var in map(abs, clause):
            break_count = compute_break(var, assignment, pos_lit_clause, neg_lit_clause)
            if break_count < min_break:
                min_break = break_count
                candidates = [var]
            elif break_count == min_break:
                candidates.append(var)

        if min_break == 0 or random.random() < p:
            var_to_flip = random.choice(candidates)
        else:
            var_to_flip = random.choice(list(map(abs, clause)))
        # Flip the chosen variable
        assignment[var_to_flip - 1] = not assignment[var_to_flip - 1]
   
    end_time = time.time()
    elapsed_time = end_time-start_time
 
    return None  # No solution found

def parse_DIMACS_to_clauses(cnf_str):
    clauses = []
    lines = cnf_str.splitlines()
    for line in lines:
        if line.startswith('p') or line.startswith('c'):
            continue
        clause = [int(lit) for lit in line.strip().split()[:-1]]
        clauses.append(clause)
    return clauses


def is_unique(solution, solutions_set):
    return tuple(solution) not in solutions_set

def generate_non_solution(num_vars):
    # Generate a random boolean assignment
    return [random.choice([True, False]) for _ in range(num_vars)]

def is_satisfied(cnf_clauses, assignment):
    for clause in cnf_clauses:
        if not any(assignment[abs(lit)-1] == (lit > 0) for lit in clause):
            return False  # Clause not satisfied
    return True  # All clauses satisfied

def parse_cnf_formula(file_path):
    cnf = []    
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()  
                if line.startswith('p'):
                    continue  
                elif line.startswith('c'):
                    continue  
                else:
                    clause_literals = [int(literal) for literal in line.split() if literal != '0']
                    cnf.append(clause_literals)
        return cnf
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def parse_solution_string(solution_string):
    if 'v' not in solution_string:
        return []
    elif(solution_string.startswith('s UNSATISFIABLE')):
        return []
    
    # Extract part after 'v' and before the last '0'
    solution_part = solution_string.split('v')[1].strip().split()[:-1]
    
    # Convert the string numbers to integers
    solution_list = [int(num) for num in solution_part]
    
    return solution_list

def process_assignments(assignments, is_solution, num_vars):
    bool_value = '1' if is_solution else '0'
    # Use space instead of a comma to separate literals
    processed_assignments = ' '.join(f'x{i + 1}: {int(assignments[i])}' for i in range(num_vars))
    return f'{processed_assignments} boolean: {bool_value}'




def read_cnf_formula(file_path):
    cnf_formula = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('p'):
                continue
            clause = ' OR '.join(f'{"-" if int(lit) < 0 else ""}x{abs(int(lit))}' for lit in line.strip().split()[:-1])
            cnf_formula.append(clause)
    return ' '.join(cnf_formula)

def create_dataset(solution_list, non_solution_list, cnf_formula):
    dataset = []
    for solution in solution_list:
        # process_assignments expects a list of True/False values
        processed_solution = process_assignments(solution[1:], True)  # Skip the first element as it's unused
        dataset.append((processed_solution, f'cnf: {cnf_formula}'))

    for non_solution in non_solution_list:
        # process_assignments expects a list of True/False values
        processed_non_solution = process_assignments([x > 0 for x in non_solution], False)
        dataset.append((processed_non_solution, f'cnf: {cnf_formula}'))

    return dataset

def format_cnf_formula(clauses):
    formatted_formula = []
    for clause in clauses:
        formatted_clause = []
        for lit in clause:
            literal = int(lit)  # Ensure the literal is an integer
            if literal < 0:
                formatted_clause.append(f"-x{abs(literal)}")
            else:
                formatted_clause.append(f"x{literal}")
        formatted_formula.append(f"({' OR '.join(formatted_clause)})")
    return ' AND '.join(formatted_formula)


def create_source_target_data(assignments_list, cnf_formulas_list):
    source_data = []
    target_data = []

    for formula_dict in assignments_list:
        formula_number = list(formula_dict.keys())[0]
        assignment_batch = formula_dict[formula_number]
        
        # Combine all assignments for a formula into a single string
        source_line = ' '.join(assignment_batch)
        source_data.append(source_line)

        # Find the corresponding CNF formula using the formula number
        cnf_formula = next(cnf_dict[formula_number] for cnf_dict in cnf_formulas_list if formula_number in cnf_dict)
        target_data.append(cnf_formula)

    return source_data, target_data


def process_formula(num_vars, num_clauses, clause_length,formula_number,instances):
    # Generate CNF formula

    logging.info(f"Starting processing formula {formula_number}")
    try:
     
        cnf_dimacs = generate_cnf_formula(num_vars, num_clauses, clause_length)
        dimacs_clauses = parse_DIMACS_to_clauses(cnf_dimacs)
        solutions = set()
        non_solutions = set()
        
        # Find solutions
        number_of_attempts = instances
   
        for _ in range(number_of_attempts):  # Number of attempts to find unique solutions
            solution = walkSAT(dimacs_clauses)
            if solution and is_satisfied(dimacs_clauses, solution) and len(solution) == num_vars:
                solution_tuple = tuple(solution)
                if is_unique(solution_tuple, solutions):
                    solutions.add(solution_tuple)


            if len(solutions) >= instances//2:
                solutions = set(random.sample(list(solutions), instances//2))


        # Generate non-solutions
        while len(non_solutions) < instances - len(solutions):
            non_solution = generate_non_solution(num_vars)
            if not is_satisfied(dimacs_clauses, non_solution) and len(non_solution) == num_vars:
                non_solution_tuple = tuple(non_solution)
                if is_unique(non_solution_tuple, non_solutions):
                    non_solutions.add(non_solution_tuple)

        cnf_formula = format_cnf_formula(dimacs_clauses)
        assignments_dict = {formula_number: [process_assignments(sol, True, num_vars) for sol in solutions] +
                                            [process_assignments(non_sol, False, num_vars) for non_sol in non_solutions]}
        cnf_formula_dict = {formula_number: format_cnf_formula(dimacs_clauses)}
        logging.info(f"Completed processing formula {formula_number}")
        return assignments_dict, cnf_formula_dict
    except Exception as e:
        print(f"Error in thread processing formula {formula_number}: {e}")
        return None

def combine_and_sort_data(assignments_list, cnf_formulas_list):
    combined_data = []
    for formula_dict in assignments_list:
        formula_number = list(formula_dict.keys())[0]
        assignment_batch = formula_dict[formula_number]
        cnf_formula = next(cnf_dict[formula_number] for cnf_dict in cnf_formulas_list if formula_number in cnf_dict)
        combined_data.append((formula_number, assignment_batch, cnf_formula))
    
    # Sort by formula number
    combined_data.sort(key=lambda x: x[0])
    return combined_data


def generate_dataset(num_vars, num_clauses,instances,DIR_PATH):
    start_time = time.time()
    logging.info('Script Started')
    clause_length = 3
    num_formulas = 100
    max_retries = 1
    balance_threshold = 0.50
    formula_stats = []
    all_assignments = []
    all_cnf_formulas = []
    all_assignments = []
    all_cnf_formulas = []
    directory_path = DIR_PATH
    overall_equal_ratio_count = 0
    overall_below_half_ratio_count = 0
    overall_above_half_ratio_count = 0
    for retry in range(max_retries):
        print(f'Starting Retry {retry}')
        retry_assignments = []
        retry_cnf_formulas = []
        solutions_count = 0
        non_solutions_count = 0
        retry_formula_stats = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_formulas) as executor:
            futures = []
            for formula_number in range(1, num_formulas + 1):
                futures.append(executor.submit(process_formula, num_vars, num_clauses, clause_length, formula_number, instances))
            for future in concurrent.futures.as_completed(futures):
                try:
                    assignments_dict, cnf_formula_dict = future.result()
                    retry_assignments.append(assignments_dict)
                    retry_cnf_formulas.append(cnf_formula_dict)
                except Exception as e:
                    print(f"Thread raised an exception: {e}")

        for assignments in retry_assignments:
            for formula_number, assignment_list in assignments.items():
                sol_count = sum('boolean: 1' in assignment for assignment in assignment_list)
                non_sol_count = sum('boolean: 0' in assignment for assignment in assignment_list)
                solutions_count += sol_count
                non_solutions_count += non_sol_count
                retry_formula_stats.append((formula_number, sol_count, non_sol_count))



        balance_ratio = solutions_count / (solutions_count + non_solutions_count) if solutions_count + non_solutions_count > 0 else 0
        if balance_ratio >= balance_threshold:
            all_formula_stats = formula_stats  # Use the stats from the current retry
            all_assignments = retry_assignments
            all_cnf_formulas = retry_cnf_formulas
            break

    equal_counts = sum(sol_count == non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)
    more_solutions = sum(sol_count > non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)
    more_non_solutions = sum(sol_count < non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)

    source_file_path = os.path.join(directory_path, f"{num_vars}v_{num_clauses}cl_source.txt")
    target_file_path = os.path.join(directory_path, f"{num_vars}v_{num_clauses}cl_target.txt")
   


    source_data, target_data = create_source_target_data(all_assignments, all_cnf_formulas)
    combined_data = combine_and_sort_data(all_assignments, all_cnf_formulas)

    with open(source_file_path, 'w') as source_file, open(target_file_path, 'w') as target_file:
        for formula_number in range(1, num_formulas + 1):
            # Find the assignments and CNF formula for the current formula number
            assignments = next((d[formula_number] for d in retry_assignments if formula_number in d), None)
            cnf_formula = next((d[formula_number] for d in retry_cnf_formulas if formula_number in d), None)

            if assignments is not None and cnf_formula is not None:
                # Write assignments and CNF formula to the files
                source_file.write(','.join(assignments) + '\n')
                target_file.write(cnf_formula + '\n')
   

    end_time = time.time()
    return source_file_path, target_file_path,instances 

def generate_predictions(num_vars, num_clauses, model_path,instances,OUTPUT_PATH,dir_path):
    source_file,target_file,instances = generate_dataset(num_vars, num_clauses,instances,dir_path)
    def load_source_data(source_file):
        with open(source_file, 'r') as src_file:
            return [line.strip() for line in src_file.readlines()]
        
    print('Script Started')
    output_path = os.path.join(OUTPUT_PATH,f"{num_vars}_{num_clauses}_pred.txt")
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

def main(num_vars, num_clauses, model_path,instances,output_path,directory_path):
    predicted_file,ground_truth_file = generate_predictions(num_vars, num_clauses, model_path,instances,output_path,directory_path)
    def read_cnf_formulas_from_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

   
    predicted_cnfs = read_cnf_formulas_from_file(predicted_file)
    ground_truth_cnfs = read_cnf_formulas_from_file(ground_truth_file)

        # Assuming all predictions are valid for simplification
    valid_predictions_count = len(predicted_cnfs)

    precision, recall, f1_score = batch_equisatisfiability_check(predicted_cnfs, ground_truth_cnfs, valid_predictions_count,alpha = 0.5)
    print(f"Precision: {round(precision,2)}")
    print(f"Recall: {round(recall,2)}")
    print(f"f1_score: {round(f1_score,2)}")
    metrics = {
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1_score, 2),
    }
    print(f"PROCESS COMPLETED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNF Formula Generator")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir_path', type=str, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)
    parser.add_argument('--num_vars', type=int, required=True)
    parser.add_argument('--num_clauses', type=int, required=True)
    parser.add_argument('--instances', type=int, required=True)

    
    args = parser.parse_args()

    main(args.num_vars, args.num_clauses,args.model_path, args.instances,args.output_dir_path,args.data_dir_path)


