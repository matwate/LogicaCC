from itertools import product
from time import time
from Queens import *


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


def main():
    formula = "AYB"
    formula_rpn = to_reverse_polish(formula)
    print(f"Formula:                            {formula}")
    print(f"Formula in Reverse Polish Notation: {formula_rpn}")
    print(f"Formula back to infix:              {to_infix_notation(formula_rpn)}")
    formula_fnc = to_fnc(formula, return_rpn=False)
    formula_fnc_rpn = to_fnc(formula, return_rpn=True)
    print(f"Formula in FNC:                     {formula_fnc}")
    print(f"Formula in FNC RPN:                 {formula_fnc_rpn}")

    formula_tseitin = tseitin_transform(formula, return_rpn=False)
    formula_tseitin_rpn = tseitin_transform(formula, return_rpn=True)
    formula_tseitin_fnc = to_fnc(formula_tseitin, return_rpn=False)
    formula_tseitin_fnc_rpn = to_fnc(formula_tseitin, return_rpn=True)
    print(f"Formula in Tseitin:                 {formula_tseitin}")
    print(f"Formula in Tseitin RPN:             {formula_tseitin_rpn}")
    print(f"Formula in Tseitin FNC:             {formula_tseitin_fnc}")
    print
    print(f"Formula in Tseitin FNC RPN:         {formula_tseitin_fnc_rpn}")


if __name__ == "__main__":
    main()
