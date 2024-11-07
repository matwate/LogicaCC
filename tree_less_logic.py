from itertools import product
from time import time
from Queens import *
from functools import reduce
import random


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
    if not return_rpn:
        return clauses
    return [to_reverse_polish(clause).replace(" ", "") for clause in clauses]


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

    def remove_literal_from_clause_list(clauses: list, literal: str, model: dict):
        unit_clause = literal
        if len(unit_clause) == 1:
            model[unit_clause] = True
        else:
            model[unit_clause[1]] = False
        for clause in clauses:
            if unit_clause in clause:
                clauses.remove(clause)
            if len(unit_clause) == 1:
                if f"-{unit_clause}" in clause:
                    clause = clause.replace(f"-{unit_clause}", "")
            else:
                if unit_clause[1] in clause:
                    clause = clause.replace(unit_clause[1], "")
        return clauses, model

    s_prime, i_prime = remove_literal_from_clause_list(S, the_choice, I)
    result, model = DPLL(s_prime, i_prime)
    if result == "Satisfacible":
        return result, model
    else:
        negated_choice = f"-{the_choice}" if not the_choice.startswith("-") else the_choice[1:]
        s_prime, i_prime = remove_literal_from_clause_list(S, negated_choice, I)
        result, model = DPLL(s_prime, i_prime)
        return result, model


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
        return any(len(clause) < 3 and len(clause) >= 1 for clause in clauses)

    def get_unit_clause(clauses: list):
        for clause in clauses:
            if len(clause) < 3 and len(clause) >= 1:
                return clause

    while not has_empty_clause(clauses) and has_unit_clause(clauses):
        unit_clause = get_unit_clause(clauses)
        if len(unit_clause) == 1:
            model[unit_clause] = True
        else:
            model[unit_clause[1]] = False
        for clause in clauses:
            if unit_clause in clause:
                clauses.remove(clause)
            if len(unit_clause) == 1:
                if f"-{unit_clause}" in clause:
                    clause = clause.replace(f"-{unit_clause}", "")
            else:
                if unit_clause[1] in clause:
                    clause = clause.replace(unit_clause[1], "")
    return clauses, model


def main():
    formula = "(AYB)O(CYD)"
    formula_tseitin = tseitin_transform(formula, return_rpn=False)
    print(formula)
    print()
    print(formula_tseitin)
    print(
        f"Unit Propagated:                   {unit_propagate(separate_by_clauses(formula_tseitin), {})}"
    )
    print(
        f"DPLL:                             {DPLL(separate_by_clauses(formula_tseitin), {})}"
    )
    print()
    print("Trying to DPLL the formula but with it in FNC")
    formula_fnc = to_fnc(formula, return_rpn=False)
    print(formula_fnc)
    print()
    print(
        f"Unit Propagated:                   {unit_propagate(separate_by_clauses(formula_fnc), {})}"
    )
    print(
        f"DPLL:                             {DPLL(separate_by_clauses(formula_fnc), {})}"
    )


if __name__ == "__main__":
    main()
