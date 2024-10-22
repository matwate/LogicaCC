from Logica import * 
from types import MethodType
from functools import reduce    
from itertools import combinations

def escribir_reinas(self, literal):
    if '-' in literal:
        atomo = literal[1:]
        neg = 'No'
    else:
        atomo = literal
        neg = ''
    x, y  = self.unravel(atomo)
    return f"{neg} {'h' if neg =='No' else 'H'}ay una reina en la casilla ({x},{y})"


class Reinas:
    def __init__(self, size: int):
        self.board_size =  size
        self.Reina_En_Casilla = Descriptor([size, size])
        self.Reina_En_Casilla.escribir = MethodType(escribir_reinas, self.Reina_En_Casilla)
        r1 = self.regla1()
        r2 = self.regla2()
        r3 = self.regla3()
        r4 = self.regla4()  
        self.reglas = [r1, r2, r3, r4]
    def regla1(self):
        regla1 = ''
        for col in range(self.board_size):
            for fila in range(self.board_size):
                casilla = (col, fila)
                other = [(col, i ) for i in range(self.board_size) if i != fila]
                formula_1 = ''
                for otra_casilla in other:
                    u = otra_casilla[0]
                    v = otra_casilla[1]
                    if not formula_1:
                        formula_1 = self.Reina_En_Casilla.ravel([u,v])
                    else:
                        formula_1 = f"({formula_1}O{self.Reina_En_Casilla.ravel([u,v])})"
                formula_1 = f"({self.Reina_En_Casilla.ravel([col, fila])}>-{formula_1})"
                if not regla1:
                    regla1 = formula_1
                else:
                    regla1 = f"({regla1}Y{formula_1})"

        return regla1
    def regla2(self):
        regla2 = ''
        for col in range(self.board_size):
            for row in range(self.board_size):
                casilla = (col, row)
                otras_casillas = [(i, row) for i in range(self.board_size) if (i, row) != casilla]
                formula1 = ''
                for otra_casilla in otras_casillas:
                    u = otra_casilla[0]
                    v = otra_casilla[1]
                    if not formula1:
                        formula1 = self.Reina_En_Casilla.ravel([u, v])
                    else:
                        formula1 = f"({formula1}O{self.Reina_En_Casilla.ravel([u, v])})"
                formula1 = f"({self.Reina_En_Casilla.ravel([col, row])}>-{formula1})"

                if not regla2:
                    regla2 = formula1
                else:
                    regla2 = f"({regla2}Y{formula1})"

        return regla2
    def regla3(self):
        regla3 = ''
        for col in range(self.board_size):
            for row in range(self.board_size):
                casilla = (col, row)
                casillas_diagonal = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if (i, j) != casilla and abs(col - i) == abs(row - j)]
                formula1 = ''
                for otra_casilla in casillas_diagonal:
                    u = otra_casilla[0]
                    v = otra_casilla[1]
                    if not formula1:
                        formula1 = self.Reina_En_Casilla.ravel([u, v])
                    else:
                        formula1 = f"({formula1}O{self.Reina_En_Casilla.ravel([u, v])})"
                formula1 = f"({self.Reina_En_Casilla.ravel([col, row])}>-{formula1})"

                if not regla3:
                    regla3 = formula1
                else:
                    regla3 = f"({regla3}Y{formula1})"

        return regla3
    def regla4(self):
        print("Building regla4...")
        n = self.board_size
        ReinaEn = self.Reina_En_Casilla
        posiciones = [(x, y) for x in range(n) for y in range(n)]
        combinaciones = list(combinations(posiciones, n))
        reglas = []
        
        for combinacion in combinaciones:
            atomos_pos = [ReinaEn.ravel([x, y]) for x, y in combinacion]
            parte_pos = reduce(lambda x, y: f"({x}Y{y})", atomos_pos)
            """
            atomos_neg = [f"-{ReinaEn.ravel(c)}" for c in posiciones if c not in combinacion]
            parte_neg = reduce(lambda x, y: f"({x}Y{y})", atomos_neg)
            regla = f"({parte_pos}Y{parte_neg})"
            reglas.append(regla)
            """
            reglas.append(parte_pos)
        
        print("regla4 built successfully")
        return reduce(lambda x, y: f"({x}O{y})", reglas)
    

def super_important_function_that_every_project_needs():
    print(
"""
  ／l、             
（ﾟ､ ｡ ７         
  l  ~ヽ       
  じしf_,)ノ        
""")
 
def main():
    r = Reinas(4)
    print(visualizar_formula(r.reglas[0], r.Reina_En_Casilla))
    print("__________")
    print(visualizar_formula(r.reglas[1], r.Reina_En_Casilla))
    print("__________")
    print(visualizar_formula(r.reglas[2], r.Reina_En_Casilla))
   
   
if __name__ == "__main__":
    main()