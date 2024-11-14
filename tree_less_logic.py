from itertools import product
from time import time
from functools import reduce
import random
from Logica import *


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.100f}s")
        return result

    return wrap_func


def to_reverse_polish(formula: str):
    precedence = {
        "Y": 2,  # AND
        "O": 1,  # OR
        ">": 0,  # IMPLIES
        "-": 3,  # NOT (highest precedence)
    }
    output = []
    operator_stack = []

    i = 0
    while i < len(formula):
        token = formula[i]
        if token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            operator_stack.pop()  # Discard the "("
        elif token in precedence:
            if token == "-" and i + 1 < len(formula) and formula[i + 1] == "-":
                # Handle double negation by appending both negations
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and precedence[operator_stack[-1]] >= precedence[token]
                ):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
                i += 1  # Move to the second negation
                operator_stack.append(token)  # Append the second negation
            else:
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and precedence[operator_stack[-1]] >= precedence[token]
                ):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
        else:
            output.append(token)
        i += 1
    while operator_stack:
        output.append(operator_stack.pop())
    return " ".join(output)


def evaluate_rpn(rpn: str, inter: dict):
    stack = []
    tokens = rpn.split()
    for token in tokens:
        if token in ["Y", "O", ">"]:
            operand2 = stack.pop()
            operand1 = stack.pop()
            if token == "Y":
                stack.append(operand1 and operand2)
            elif token == "O":
                stack.append(operand1 or operand2)
            else:
                stack.append(not operand1 or operand2)
        elif token == "-":
            operand = stack.pop()
            stack.append(not operand)
        else:
            stack.append(inter[token])
    return stack.pop()


def letras_proposicionales(formula):
    return sorted(
        set(char for char in formula if char not in ["Y", "O", ">", "(", ")", "-"])
    )


@timer
def SATtabla(formula: str):
    rpn = to_reverse_polish(formula)
    letras = letras_proposicionales(formula)
    n = len(letras)
    valores = list(product([True, False], repeat=n))
    for i in valores:
        inter = dict(zip(letras, i))
        if evaluate_rpn(rpn, inter):
            return inter
    return None


def clasificar_tableaux(rpn: str):
    """
    Esta funcion determina si una funcion es alfa, beta o literal
    De tal forma que se pueda clasificar en un tableau
    Alfa: Alfa(RPN)
    - (A Y B): A B Y
    - --A : A - -
    - -(A O B) : A B O -
    - -(A > B) : A B > -

    Beta: Beta(RPN)
    - (A O B): A B O
    - -(A Y B) : A B Y -
    - (A > B) : A B >

    Por lo que si revisamos los ultimos 2 elementos de la formula en RPN
    podemos determinar si es alfa o beta
    """

    rpn = rpn.replace(" ", "")

    if len(rpn) < 2:
        return "Literal"

    last = rpn[-1]
    penultimate = rpn[-2]

    if last in "-":
        if penultimate in "O>-":
            return "Alfa"
        elif penultimate in "Y":
            return "Beta"
        else:
            return "Literal"
    else:
        if last in "O>":
            return "Beta"
        elif last in "Y":
            return "Alfa"
        else:
            return "Literal"


def to_infix_notation(rpn_expression: str):
    stack = []
    operators = set(["Y", "O", ">", "-"])
    tokens = rpn_expression.split()

    for token in tokens:
        if token not in operators:
            stack.append(token)
        elif token == "-":
            # Handle negation, including double negation
            neg_count = 1
            while stack and stack[-1] == "-":
                neg_count += 1
                stack.pop()

            if stack:
                operand = stack.pop()
                negation = "-" * neg_count
                stack.append(f"({negation}{operand})")
            else:
                stack.append("-" * neg_count)
        else:
            right = stack.pop()
            left = stack.pop()
            if token == "Y":
                stack.append(f"({left}Y{right})")
            elif token == "O":
                stack.append(f"({left}O{right})")
            elif token == ">":
                stack.append(f"({left}>{right})")

    return stack[0]


def find_subf(rpn: str, return_rpn=False):
    # Esta funcion encuentra las subformulas de una formula
    """
    La manera en la que la pense, retorna las subformulas en infix y
    no en notacion polaca inversa, pero creo que es util de todas formas

    va a hacer reconstruccion de la formula en notacion infija de todo menos el
    ultimo operando.
    """
    stack = []
    tokens = rpn.replace(" ", "")
    for i, token in enumerate(tokens):
        if i == len(tokens) - 1:
            break
        if token in "YO>":
            operand2 = stack.pop()
            operand1 = stack.pop()
            stack.append(f"({operand1}{token}{operand2})")
        elif token == "-":
            operand = stack.pop()
            stack.append(f"-{operand}")
        else:
            stack.append(token)

    return stack if not return_rpn else [to_reverse_polish(sub) for sub in stack]


def find_conectives(rpn: str):
    if len(rpn) < 2:
        return None
    return rpn[-1] if rpn[-1] in "YO>" else rpn[-2]


def is_negated(rpn: str):
    return rpn[-1] == "-"


def expandir_para_tableaux(rpn: str, return_rpn=False):
    """
    Esta funcion expande un nodo de un tableau con esta regla
    - Si es alfa, se colocan ambas subformulas en el mismo conjunto
    - Si es beta, se colocan en conjuntos separados
    - Si es literal, no se hace nada
    """
    rpn = rpn.replace(" ", "")
    clasf = clasificar_tableaux(rpn)
    subfs = find_subf(rpn, return_rpn=True)
    output = []
    match clasf:
        case "Alfa":
            if is_negated(rpn):
                match find_conectives(rpn):
                    case "O":
                        output.extend([f"{sub}-" for sub in subfs])
                    case ">":
                        subfs = find_subf(rpn[:-1], return_rpn=True)
                        output.extend([subfs[0], f"{subfs[1]}-"])
                if is_negated(subfs[0]):
                    output.extend(
                        [f"{sub}" for sub in find_subf(subfs[0], return_rpn=True)]
                    )
            else:
                match find_conectives(rpn):
                    case "Y":
                        output.extend([f"{sub}" for sub in subfs])
        case "Beta":
            if is_negated(rpn):
                match find_conectives(rpn):
                    case "Y":
                        output.extend([[f"{sub}-"] for sub in subfs])
                if is_negated(subfs[0]):
                    output.extend(
                        [f"{sub}" for sub in find_subf(subfs[0], return_rpn=True)]
                    )
            else:
                match find_conectives(rpn):
                    case "O":
                        output.extend([[f"{sub}"] for sub in subfs])
                    case ">":
                        output.extend([[f"{subfs[0]}-"], [f"{subfs[1]}"]])
        case "Literal":
            output.append([rpn])
    return output


# I no longer care about tablaux.


def to_fnc(formula: str, return_rpn=False):
    # Esta funcion transforma una formula a su forma normal conjuntiva
    rpn = to_reverse_polish(formula)

    # Se va a ir evaluando la formula en notacion polaca inversa, aplicando las reglas a medida que se construye
    stack = []
    tokens = rpn.split(" ")
    for token in tokens:
        match token:
            case ">":
                right = stack.pop()
                left = stack.pop()
                stack.append(f"(-{left}O{right})")
            case "Y":
                right = stack.pop()
                left = stack.pop()
                stack.append(f"({left}Y{right})")
            case "O":
                right = stack.pop()
                left = stack.pop()
                left_con = find_conectives(to_reverse_polish(left))
                right_con = find_conectives(to_reverse_polish(right))
                if left_con == "Y" and right_con == "Y":
                    left = to_reverse_polish(left)
                    right = to_reverse_polish(right)
                    left_subfs = find_subf(left)
                    l = left_subfs[0]
                    r = left_subfs[1]
                    right_subfs = find_subf(right)
                    l2 = right_subfs[0]
                    r2 = right_subfs[1]
                    stack.append(f"({l}O{l2})Y({l}O{r2})Y({r}O{l2})Y({r}O{r2})")
                elif left_con == "Y":
                    left = to_reverse_polish(left)
                    left_subfs = find_subf(left)
                    l = left_subfs[0]
                    r = left_subfs[1]
                    stack.append(f"({l}O{right})Y({r}O{right})")
                elif right_con == "Y":
                    right = to_reverse_polish(right)
                    right_subfs = find_subf(right)
                    l = right_subfs[0]
                    r = right_subfs[1]
                    stack.append(f"({left}O{l})Y({left}O{r})")
                else:
                    stack.append(f"({left}O{right})")
            case "-":
                operand = stack.pop()
                operand_con = find_conectives(to_reverse_polish(operand))
                if len(operand) == 1:
                    stack.append(f"-{operand}")
                elif operand_con == "Y":
                    subfs = find_subf(operand)
                    l = subfs[0]
                    r = subfs[1]
                    stack.append(f"(-{l}O-{r})")
                elif operand_con == "O":
                    op_clasf = clasificar_tableaux(operand)
                    if op_clasf == "Alfa" or op_clasf == "Beta":
                        subfs = find_subf(operand)
                        l = subfs[0]
                        r = subfs[1]
                        stack.append(f"(-{l}Y-{r})")
                    else:
                        stack.append(f"-{operand}")
                elif operand_con == ">":
                    subfs = find_subf(operand)
                    l = subfs[0]
                    r = subfs[1]
                    stack.append(f"({l}Y-{r})")
                else:
                    stack.append(f"-{operand}")
            case _:
                stack.append(token)
    result = stack[0]
    return result if not return_rpn else to_reverse_polish(result)


def tseitin_transform(formula: str, return_rpn: bool = False):
    """
    Esta fuuncion transforma una formula a su transformacion de Tseitin:

    Proceso:
    Para cada subformula de la formula se le asigna una variable.
    Luego se reemplazan las subformulas por las variables asignadas.
    Se construye una formula que es equivalente a la original a modo que cada variable este <> a la subformula que reemplaza.

    Ejemplo:
    (A Y B) O C

    Se asignan variables a cada subformula

    A Y B = X1
    X1 O C = X2

    Se construye la formula equivalente
    (X1 <> (A Y B)) Y (X2 <> (X1 O C)) Y X2
    """
    rpn = to_reverse_polish(formula)
    letras_chars = [ord(char) for char in letras_proposicionales(formula)]
    var_min_char = max(letras_chars) + 256

    # La forma en la que se hace es mediante evaulando la formula en notacion polaca inversa
    # A modo de construir la formula original pero remplazando por lo que en verdad necesitamos
    stack = []
    form_stack = (
        []
    )  # Necesitamos 2 stacks porque tenemos que trackear las subformulas + las variables, esta va a dar el resultado final
    tokens = rpn.split(" ")
    variables_counter = 1
    for token in tokens:
        match token:
            case ">":
                var = chr(var_min_char + variables_counter)
                variables_counter += 1
                if var in "YO>":
                    var = chr(var_min_char + variables_counter + 1)
                right = stack.pop()
                left = stack.pop()
                stack.append(var)
                form_stack.append(
                    f"({left}O{var})Y(-{right}O{var})Y(-{left}O{right}O-{var})"
                )  # FNC de p <> (q > r)
            case "Y":
                var = chr(var_min_char + variables_counter)
                variables_counter += 1
                if var in "YO>":
                    var = chr(var_min_char + variables_counter + 1)
                right = stack.pop()
                left = stack.pop()
                stack.append(var)
                form_stack.append(
                    f"({left}O-{var})Y({right}O-{var})Y(-{left}O-{right}O{var})"
                )  # FNC de p <> (q Y r)

            case "O":
                var = chr(var_min_char + variables_counter)
                variables_counter += 1
                if var in "YO>":
                    var = chr(var_min_char + variables_counter + 1)
                right = stack.pop()
                left = stack.pop()
                stack.append(var)
                form_stack.append(
                    f"((-{left}O{var})Y(-{right}O{var})Y({left}O{right}O-{var}))"
                )  # FNC de p <> (q O r)

            case "-":
                var = chr(var_min_char + variables_counter)
                variables_counter += 1
                if var in "YO>":
                    var = chr(var_min_char + variables_counter + 1)
                operand = stack.pop()
                stack.append(var)
                form_stack.append(
                    f"((-{var}O-{operand})Y({var}O{operand}))"
                )  # FNC de p <> -q

            case _:
                stack.append(token)
    last_var = stack[0]
    form_stack.append(last_var)
    return (
        reduce(lambda x, y: f"{x}Y{y}", form_stack)
        if not return_rpn
        else to_reverse_polish(reduce(lambda x, y: f"{x}Y{y}", form_stack))
    )


def separate_by_clauses(formula: str, return_rpn: bool = False):
    clauses = [
        clause.replace("(", "").replace(")", "") for clause in formula.split("Y")
    ]
    if return_rpn:
        return [to_reverse_polish(clause) for clause in clauses]
    return clauses


def DPLL(clauses: list, model: dict):
    # Despite me being the "DO NOT RECURSIVE AAAAAAAA IT WILL KILL YOU" in this case is valid
    # Because the max depth on the formulas im going to be using it is max 64 or 128 smthn so
    # It wont eat my whole RAM

    # Unit propagate the whole thing
    S, I = unit_propagate(clauses, model)
    if has_empty_clause(S):
        return "Insatisfacible", {}
    if len(S) == 0:
        return "Satisfacible", I
    all_literals = all_literals_of_clauses(S)
    the_choice = random.choice(list(all_literals))

    s_prime, i_prime = remove_literal_from_clause_list(S, the_choice, I)
    result, model = DPLL(s_prime, i_prime)
    if result == "Satisfacible":
        return result, model
    else:
        negated_choice = (
            f"-{the_choice}" if not the_choice.startswith("-") else the_choice[1:]
        )
        s_prime, i_prime = remove_literal_from_clause_list(S, negated_choice, I)
        result, model = DPLL(s_prime, i_prime)
        return result, model


def remove_literal_from_clause_list(clauses: list, literal: str, model: dict):
    # Remove clauses with the literal as is.
    new_clauses = []
    for clause in clauses:
        match len(literal):
            case 1:
                if literal in clause and f"-{literal}" not in clause:
                    model[literal] = True
                    continue
                if f"-{literal}" in clause:
                    new_clause = clause.replace(f"-{literal}O", "").replace(
                        f"O-{literal}", ""
                    )
                    new_clauses.append(new_clause)
                else:
                    new_clauses.append(clause)
            case 2:
                if literal in clause:
                    model[literal[1]] = False
                    continue
                if literal[1] in clause:
                    new_clause = clause.replace(f"{literal[1]}O", "").replace(
                        f"O{literal[1]}", ""
                    )
                    new_clauses.append(new_clause)
    print(new_clauses)
    return new_clauses, model


def all_literals_of_clauses(clauses: list):
    literals: set[str] = set()
    for clause in clauses:
        this_literals = set(clause.split("O"))
        literals = literals.union(this_literals)
    return literals


def has_empty_clause(clauses: list):
    return any(clause == "" for clause in clauses)


def unit_propagate(clauses: list, model: dict):
    # Istg clauses should be in infix with parnthesis removed so i can do some quirky bullshit
    # That way we're going to ASSUME that the clauses are in infix notation with parenthesis so i can do stuff with
    # string manipulation

    def has_unit_clause(clauses: list):
        # A unit clause is either "x" or "-x", a non unitary clause will be AT LEAST "Xoy" so if its 2 or 1 long
        # then its an unit clause
        return any(len(clause) in [1, 2] for clause in clauses)

    def get_unit_clause(clauses_list: list):
        for clause in clauses_list:
            if len(clause) in [1, 2]:
                return clause

    while not has_empty_clause(clauses) and has_unit_clause(clauses):
        unit_clause = get_unit_clause(clauses)
        clauses, model = remove_literal_from_clause_list(clauses, unit_clause, model)
    return clauses, model


def bogo_SAT(formula: str):
    rpn = to_reverse_polish(formula)
    letras = letras_proposicionales(formula)
    n = len(letras)
    valores = list(product([True, False], repeat=n))
    random.shuffle(valores)
    for i in valores:
        inter = dict(zip(letras, i))
        if evaluate_rpn(rpn, inter):
            return inter
    return None


class WalkSat:

    def __init__(self, formula: str, max_flips: int, max_tries: int, temp: float):
        self.formula = formula
        self.rpn = to_reverse_polish(formula)
        self.letras = letras_proposicionales(tseitin_transform(formula))
        self.n = len(self.letras)
        self.clause_list = separate_by_clauses(tseitin_transform(self.formula), False)
        self.clause_list_rpn = separate_by_clauses(
            tseitin_transform(self.formula, False), True
        )
        self.clause_satisfied: list[bool] = []

        self.max_flips = max_flips
        self.max_tries = max_tries
        self.model = {p: random.choice([True, False]) for p in self.letras}
        self.temp = temp
        self.literals = all_literals_of_clauses(self.clause_list)

    def try_(self):
        # This will be a try of the algo
        # 1. Eval the formula with the model
        sat = evaluate_rpn(self.rpn, self.model)
        if sat:
            return "SAT", self.model
        for _ in range(self.max_flips):
            lit = self.least_break_force()
            will_it_flip = random.uniform(0, 1)
            if will_it_flip > self.temp:
                self.model = self.flip_literal(lit, self.model)
            else:
                # Pick a random literal
                self.model = self.flip_literal(
                    random.choice(list(self.literals)), self.model
                )
            # Evaluate the formula with the new model
            sat = evaluate_rpn(self.rpn, self.model)
            if sat:
                return "SAT", self.model
        return "IDK YET", self.model

    def least_break_force(self):
        # This function will return the literal that will break the least amount of clauses
        # 1. Evaluate the clauses with the current model
        clause_satisfied = [
            evaluate_rpn(clause, self.model) for clause in self.clause_list_rpn
        ]
        # 2. For every literal in the model, check if flipping it will break the least amount of clauses
        min_break_force = 10**10
        min_break_literal = None
        for lit in list(self.literals):
            temp_model = self.flip_literal(lit, self.model)
            temp_clause_satisfied = [
                evaluate_rpn(clause, temp_model) for clause in self.clause_list_rpn
            ]
            # Element by element comparison of the clause sat and temp clause sat to check if any true in the first is false in the other
            break_force = sum(
                [
                    1
                    for i, j in zip(clause_satisfied, temp_clause_satisfied)
                    if i and not j
                ]
            )
            if break_force < min_break_force:
                min_break_force = break_force
                min_break_literal = lit
        return min_break_literal

    def flip_literal(self, literal: str, model: dict):
        flipped_model = model
        let = literal[-1]
        flipped_model[let] = not model[let]
        return flipped_model

    def SAT(self):
        sat = ""
        for _ in range(self.max_tries):
            sat = self.try_()
        return sat

    def SAT_till_SAT(self):
        sat, inter = self.SAT()
        walk_sat_attempts = 1
        while sat != "SAT":
            walk_sat_attempts += 1
            sat, inter = self.SAT()
        return sat, inter


"""
    Quick todo list for when i open this again
    1. DPLL Is not working should:
        - Remove the O from the clauses
        - Properly remove negations and shit
        - Move to separate function the removal part for reusability
"""


def walkSAT(A, max_flips=10, max_tries=10, p=0.2):
    w = WalkSatEstado(A)
    for i in range(max_tries):
        w.actualizar(interpretacion_aleatoria(w.letrasp))
        for j in range(max_flips):
            if w.SAT():
                return "Satisfacible", w.I
            c = choice(w.clausulas_unsat)
            breaks = [(l, w.break_count(l)) for l in C]
            if any((x[1] == 0 for x in breaks)) > 0:
                v = choice(breaks)[0]
            else:
                if uniform(0, 1) < p:
                    assert (len(C) > 0, f"{C}")
                    v = choice(C)
                else:
                    min_count = min([x[1] for x in breaks])
                    mins = [x[0] for x in breaks if x[1] == min_count]
                    v = choice(mins)
            I = flip_literal(w.I, v)
            w.actualizar(I)
    print("Intento Fallido")
    return None


def main():
    formula = "(AYB)O(CYD)"
    w = WalkSat(formula, 10, 1, 0.5)
    print(w.SAT())


if __name__ == "__main__":
    main()
