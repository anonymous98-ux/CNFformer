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
import re
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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

def process_assignments(assignments, is_solution, num_vars):
    bool_value = '1' if is_solution else '0'
    # Use space instead of a comma to separate literals
    processed_assignments = ' '.join(f'x{i + 1}: {int(assignments[i])}' for i in range(num_vars))
    return f'{processed_assignments} boolean: {bool_value}'

def parse_clauses(cnf_string):
    # Regular expression to match clauses
    clause_pattern = re.compile(r'\((.*?)\)')
    literal_pattern = re.compile(r'(-?x\d+)')
    
    # Find all clauses
    clauses = clause_pattern.findall(cnf_string)
    parsed_clauses = []
    
    for clause in clauses:
        literals = literal_pattern.findall(clause)
        # Convert literals to integers and remove 'x' prefix
        parsed_clause = [int(lit.replace('x', '')) if 'x' in lit else -int(lit.replace('-x', '')) for lit in literals]
        parsed_clauses.append(parsed_clause)
    
    return parsed_clauses
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

def process_formula(cnf_formula, num_vars,formula_number,instances):
    logging.info(f"Starting processing formula {formula_number}")
    try:

        dimacs_clauses = parse_clauses(cnf_formula)
        solutions = set()
        non_solutions = set()
        count = 0
        # Find solutions
        number_of_attempts = instances
        for _ in range(number_of_attempts):  # Number of attempts to find unique solutions
            solution = walkSAT(dimacs_clauses)
            if solution and is_satisfied(dimacs_clauses, solution) and len(solution) == num_vars:
                solution_tuple = tuple(solution)
                if is_unique(solution_tuple, solutions):
                    count+=1
                    solutions.add(solution_tuple)


            if len(solutions) >= instances//2:
                solutions = set(random.sample(list(solutions), instances//2))
                break

        # Generate non-solutions
        while len(non_solutions) < instances - len(solutions):
            non_solution = generate_non_solution(num_vars)
            if not is_satisfied(dimacs_clauses, non_solution) and len(non_solution) == num_vars:
                non_solution_tuple = tuple(non_solution)
                if is_unique(non_solution_tuple, non_solutions):
                    non_solutions.add(non_solution_tuple)


        assignments_dict = {formula_number: [process_assignments(sol, True, num_vars) for sol in solutions] +
                                            [process_assignments(non_sol, False, num_vars) for non_sol in non_solutions]}
        cnf_formula_dict = {formula_number: format_cnf_formula(dimacs_clauses)}
       
        logging.info(f"Completed processing formula {formula_number}")
        return assignments_dict, cnf_formula_dict
    except Exception as e:
        print(f"Error in thread processing formula {formula_number}: {e}")
        return None
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
def read_cnf_from_file(file_path):
    """Read CNF from a file and return as a string."""
    with open(file_path, 'r') as file:
        cnf_string = file.read().strip()
    return cnf_string
def find_max_variable(cnf_formula):
    # Use regex to find all occurrences of variable names
    variables = re.findall(r'x(\d+)', cnf_formula)
    # Convert the extracted variable names to integers
    variable_indices = list(map(int, variables))

    return max(variable_indices)
def main(input_file,output_source,output_target):
    assignments_dict = {}
    cnf_formula_dict = {}
    cnf_formula = read_cnf_from_file(input_file)   
    num_vars = find_max_variable(cnf_formula) 
    formula_number = 1
    instances = 150
    solutions_count = 0
    non_solutions_count = 0
    retry_assignments = []
    retry_cnf_formulas = []
    retry_formula_stats = []
    assignments_dict,cnf_formula_dict = process_formula(cnf_formula,num_vars,formula_number,instances)
    retry_assignments.append(assignments_dict)
    retry_cnf_formulas.append(cnf_formula_dict)    
    for assignments in retry_assignments:
        for formula_number, assignment_list in assignments.items():
            sol_count = sum('boolean: 1' in assignment for assignment in assignment_list)
            non_sol_count = sum('boolean: 0' in assignment for assignment in assignment_list)
            solutions_count += sol_count
            non_solutions_count += non_sol_count
            retry_formula_stats.append((formula_number, sol_count, non_sol_count))
    equal_counts = sum(sol_count == non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)
    more_solutions = sum(sol_count > non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)
    more_non_solutions = sum(sol_count < non_sol_count for _, sol_count, non_sol_count in retry_formula_stats)

    source_data, target_data = create_source_target_data(retry_assignments, retry_cnf_formulas)
    combined_data = combine_and_sort_data(retry_assignments, retry_cnf_formulas)
    with open(output_source, 'w') as source_file, open(output_target, 'w') as target_file:
        # Find the assignments and CNF formula for the current formula number
        assignments = next((d[formula_number] for d in retry_assignments if formula_number in d), None)
        cnf_formula = next((d[formula_number] for d in retry_cnf_formulas if formula_number in d), None)

        if assignments is not None and cnf_formula is not None:
            # Write assignments and CNF formula to the files
            source_file.write(','.join(assignments) + '\n')
            target_file.write(cnf_formula + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNF Formula Generator")
    parser.add_argument("--input",type =str, required=True, help = "Formula Identifier for the Circuit/Problem")
    parser.add_argument("--output_source",type =str, required=True, help = "Formula Identifier for the Circuit/Problem")
    parser.add_argument("--output_target",type =str, required=True, help = "Formula Identifier for the Circuit/Problem")
    args = parser.parse_args()

    main(args.input,args.output_source,args.output_target)
