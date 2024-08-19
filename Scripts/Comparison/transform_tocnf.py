from sympy.logic.boolalg import Or, And, Not
from sympy import symbols, to_cnf
import re
import argparse

def parse_expression(expr, sympy_vars):
    """
    Parses the logical expression into sympy-compatible format.
    Assumes expr is a string representation of a logical expression.
    """
    # Replace logical operators with sympy equivalents
    expr = expr.replace('and', '&')
    expr = expr.replace('or', '|')
    expr = expr.replace('not', '~')
    
    # Evaluate the expression to convert it to sympy format
    return eval(expr, {}, sympy_vars)
def write_to_file(filename, content):


    with open(filename, 'w') as file:
        file.write(content)
def main(input_file_path,output_file_path):
    with open(input_file_path, 'r') as file:
        logical_expression = file.read().strip()
    
    # Define logical variables
    variables = re.findall(r'x_\d+', logical_expression)
    unique_vars = set(variables)
    sympy_vars = {var.replace('_', ''): symbols(var.replace('_', '')) for var in unique_vars}
    
    # Replace variable names in the expression
    for var in unique_vars:
        logical_expression = logical_expression.replace(var, var.replace('_', ''))
    
    # Create a dictionary for eval context
    eval_context = {var.replace('_', ''): sympy_vars[var.replace('_', '')] for var in unique_vars}
    
    # Parse the expression
    parsed_expr = parse_expression(logical_expression, eval_context)
    
    # Convert the expression to CNF
    cnf_expr = to_cnf(parsed_expr, simplify=True,force=True)
    cnf_str = str(cnf_expr)
    cnf_str = cnf_str.replace('~', '-').replace('&', 'AND').replace("|","OR")

    def increment_variable(match):
        var = match.group(0)
        if var.startswith('-x'):
            return f'-x{int(var[2:]) + 1}'
        else:
            return f'x{int(var[1:]) + 1}'

    # Replace variables
    cnf_str = re.sub(r'-?x\d+', increment_variable, cnf_str)
    write_to_file(output_file_path, cnf_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNF Formula Generator")
    parser.add_argument('--input', type=str, required=True, help="Number of variables in the CNF formulas")
    parser.add_argument('--output', type=str, required=True, help="Number of clauses in the CNF formulas")
    args = parser.parse_args()
    main(args.input,args.output)
