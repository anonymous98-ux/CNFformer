import random

class Clause:
    """A Boolean clause randomly generated"""

    def __init__(self, num_vars, clause_length):
        self.length = clause_length
        self.lits = []
        self.gen_random_clause(num_vars)

    def gen_random_clause(self, num_vars):
        while len(self.lits) < self.length:
            new_lit = random.randint(1, num_vars)
            if new_lit not in self.lits and -new_lit not in self.lits:
                if random.random() < 0.5:
                    new_lit = -new_lit
                self.lits.append(new_lit)

    def __str__(self):
        return " ".join(map(str, self.lits)) + " 0"

class CNF:
    """A CNF formula randomly generated"""

    def __init__(self, num_vars, num_clauses, clause_length):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clause_length = clause_length
        self.clauses = [Clause(num_vars, clause_length) for _ in range(num_clauses)]
    def verify_formula(self):
        all_vars = set(range(1, self.num_vars + 1))
        vars_in_formula = set()
        for clause in self.clauses:
            vars_in_clause = set(abs(lit) for lit in clause.lits)
            vars_in_formula.update(vars_in_clause)
        missing_vars = all_vars - vars_in_formula
        if missing_vars:
            print(f"Warning: Variables {missing_vars} are missing in the formula.")
    def __str__(self):
        formula = "c Random CNF formula\n"
        formula += f"p cnf {self.num_vars} {self.num_clauses}\n"
        formula += "\n".join(str(clause) for clause in self.clauses)
        return formula

def generate_cnf_formula(num_vars, num_clauses, clause_length):
    cnf_formula = CNF(num_vars, num_clauses, clause_length)
    return str(cnf_formula)


