import random
import time
import re
import itertools

def cnf_batch_to_list(cnf_strings):
    batch_cnf_list = []
    for cnf_string in cnf_strings:
        batch_cnf_list.append(cnf_to_list(cnf_string))
    return batch_cnf_list

def is_valid_cnf(cnf_formula):
    valid_literal_pattern = re.compile(r'(^|\s)-?x\d+(\s|$)')
    for clause in cnf_formula.split(' AND '):
        clause = clause.strip().replace('(', '').replace(')', '')
        literals = clause.split(' OR ')
        for literal in literals:
            if not valid_literal_pattern.fullmatch(literal.strip()):
                return False
    return True

def cnf_to_list(cnf_string):
    clauses = cnf_string.split(' AND ')
    cnf_list = []

    valid_literal_pattern = re.compile(r'-?x\d+')

    for clause in clauses:
        literals = clause.replace('(', '').replace(')', '').split(' OR ')
        int_literals = []

        for literal in literals:
            literal = literal.strip()
            if valid_literal_pattern.match(literal):
                try:
                    if literal.startswith('-x'):
                        int_literals.append(-int(literal.lstrip('-x')))
                    elif literal.startswith('x'):
                        int_literals.append(int(literal.lstrip('x')))
                except ValueError:
                    print(f"Value error with literal: {literal}")
            else:
                print(f"Skipping malformed literal: {literal}")

        if int_literals:
            cnf_list.append(int_literals)

    return cnf_list

def reconstruct_cnf_string(clauses):
    def format_literal(lit):
        prefix = '-' if lit < 0 else ''
        return f'{prefix}x{abs(lit)}'

    cnf_str = ' AND '.join(
        '(' + ' OR '.join(format_literal(lit) for lit in clause) + ')'
        for clause in clauses
    )
    return cnf_str

def walkSAT(clauses, p=0.5, max_flips=1000):
    start_time = time.time()
    def compute_break(variable, assignment, pos_lit_clause, neg_lit_clause):
        break_count = 0
        clauses_to_check = pos_lit_clause[variable] if assignment[variable - 1] else neg_lit_clause[variable]
        for clause_index in clauses_to_check:
            if true_literal_count(clauses[clause_index], assignment) == 1:
                break_count += 1
        return break_count

    def true_literal_count(clause, assignment):
        return sum(assignment[abs(lit) - 1] == (lit > 0) for lit in clause)

    n_vars = max(abs(lit) for clause in clauses for lit in clause)
    assignment = [random.choice([False, True]) for _ in range(n_vars)]

    pos_lit_clause = [[] for _ in range(n_vars + 1)]
    neg_lit_clause = [[] for _ in range(n_vars + 1)]

    for i, clause in enumerate(clauses):
        for lit in clause:
            if lit > 0:
                pos_lit_clause[lit].append(i)
            else:
                neg_lit_clause[abs(lit)].append(i)

    for _ in range(max_flips):
        unsatisfied = [clause for clause in clauses if not true_literal_count(clause, assignment)]
        if not unsatisfied:
            return assignment  # SATISFIABLE

        clause = random.choice(unsatisfied)
        min_break = float('inf')
        candidates = []
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
        assignment[var_to_flip - 1] = not assignment[var_to_flip - 1]

    end_time = time.time()
    elapsed_time = end_time - start_time
    return None  # No solution found

def find_satisfying_assignments(clauses, max_tries=200, max_flips=1000):
    satisfying_assignments = []
    for _ in range(max_tries):
        if len(satisfying_assignments) >= 100:
            break
        assignment = walkSAT(clauses, max_flips=max_flips)
        if assignment is not None and assignment not in satisfying_assignments:
            satisfying_assignments.append(assignment)
    return satisfying_assignments

def is_cnf_satisfied(clauses, assignment):
    for clause in clauses:
        if not any((lit > 0 and assignment[abs(lit) - 1]) or (lit < 0 and not assignment[abs(lit) - 1]) for lit in clause):
            return False
    return True

def check_equisatisfiability(ground_truth_clauses, predicted_clauses, max_tries=100, max_flips=10000):
    gt_satisfying_assignments = find_satisfying_assignments(ground_truth_clauses, max_tries, max_flips)
    for assignment in gt_satisfying_assignments:
        if not is_cnf_satisfied(predicted_clauses, assignment):
            return False
    return True if gt_satisfying_assignments else False

def check_common_assignments(gt_satisfying_assignments, predicted_clauses):
    common_satisfying_count = 0
    total_assignments = len(gt_satisfying_assignments)
    for assignment in gt_satisfying_assignments:
        if is_cnf_satisfied(predicted_clauses, assignment):
            common_satisfying_count += 1
    return common_satisfying_count, total_assignments

def extend_assignments(assignments, additional_vars):
    extended_assignments = []
    for assignment in assignments:
        for extra_vals in itertools.product([False, True], repeat=additional_vars):
            extended_assignments.append(assignment + list(extra_vals))
    return extended_assignments

def adjust_assignments_for_variable_length(predicted_satisfying_assignments, additional_vars):
    extended_assignments = extend_assignments(predicted_satisfying_assignments, additional_vars)
    return extended_assignments

def max_var_index(cnf):
    max_index = 0
    for clause in cnf:
        for literal in clause:
            max_index = max(max_index, abs(literal))
    return max_index

def filter_assignments(assignments, original_clauses):
    return [assignment for assignment in assignments if is_cnf_satisfied(original_clauses, assignment)]

def compute_precision_recall(predicted_clauses, gt_clauses, max_tries=100, max_flips=1000):
    predicted_satisfying_assignments = find_satisfying_assignments(predicted_clauses, max_tries, max_flips)
    original_length = len(predicted_satisfying_assignments)
    gt_satisfying_assignments = find_satisfying_assignments(gt_clauses, max_tries, max_flips)

    max_var_gt = max_var_index(gt_clauses)
    max_var_predicted = max_var_index(predicted_clauses)
    
    if max_var_gt > max_var_predicted:
        additional_vars = max_var_gt - max_var_predicted
        predicted_satisfying_assignments = adjust_assignments_for_variable_length(predicted_satisfying_assignments, additional_vars)
        predicted_satisfying_assignments = filter_assignments(predicted_satisfying_assignments, predicted_clauses)
    elif max_var_predicted > max_var_gt:
        additional_vars = max_var_predicted - max_var_gt
        gt_satisfying_assignments = adjust_assignments_for_variable_length(gt_satisfying_assignments, additional_vars)
        gt_satisfying_assignments = filter_assignments(gt_satisfying_assignments, gt_clauses)

    tp, fp, fn = 0, 0, 0
    unsatisfied_clauses_fp, unsatisfied_clauses_fn = 0, 0
    
    for assignment in predicted_satisfying_assignments:
        if is_cnf_satisfied(gt_clauses, assignment):
            tp += 1
        else:
            fp += 1
           

    for assignment in gt_satisfying_assignments:
        if is_cnf_satisfied(predicted_clauses, assignment):
            tp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall

def batch_equisatisfiability_check(predicted_cnfs, ground_truth_cnfs, valid_predictions_count, alpha=0.5, beta=0.5):
    total_precision = 0.0  
    total_recall = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    predicted_lists = cnf_batch_to_list(predicted_cnfs)
    ground_truth_lists = cnf_batch_to_list(ground_truth_cnfs)

    for i, (pred_list, gt_list) in enumerate(zip(predicted_lists, ground_truth_lists)):
        precision, recall = compute_precision_recall(pred_list, gt_list)

        total_precision += precision
        total_recall += recall



    avg_precision = total_precision / valid_predictions_count if valid_predictions_count else 0
    avg_recall = total_recall / valid_predictions_count if valid_predictions_count else 0
    
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0
    return avg_precision, avg_recall, f1_score
