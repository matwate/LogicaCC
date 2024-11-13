from Logica import *
from types import MethodType
from functools import reduce
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns
from PIL import Image

def escribir_reinas(self, literal):
    if "-" in literal:
        atomo = literal[1:]
        neg = "No"
    else:
        atomo = literal
        neg = ""
    x, y = self.unravel(atomo)
    return f"{neg} {'h' if neg =='No' else 'H'}ay una reina en la casilla ({x},{y})"


class Reinas:
    def __init__(self, size: int):
        self.board_size = size
        self.Reina_En_Casilla = Descriptor([size, size])
        self.Reina_En_Casilla.escribir = MethodType(
            escribir_reinas, self.Reina_En_Casilla
        )
        self.r1 = self.regla1()
        self.r2 = self.regla2()
        self.r3 = self.regla3()
        self.r4 = self.regla4()
        self.reglas = [self.r1, self.r2, self.r3, self.r4]

    def regla1(self):
        """
        Regla 1: Por cada casilla del tablero, se tiene que cumplir que o bien no hay una reina en esa casilla o bien no hay una reina en la misma columna.

        La regla se escribe como: para cada casilla (x,y) del tablero, se tiene que ((no hay una reina en (x,y)) o (no hay una reina en la columna y)) para cada columna y diferente de y.

        Se implementa como: para cada casilla (x,y) del tablero, se construye una formula que es la disyuncin de (no hay una reina en (x,y)) y la negacion de la existencia de una reina en la columna y, para cada fila y diferente de y. Luego se construye la conjuncion de todas estas formulas.
        """
        regla1 = ""
        for col in range(self.board_size):
            for fila in range(self.board_size):
                other = [(col, i) for i in range(self.board_size) if i != fila]
                formula_1 = ""
                for otra_casilla in other:
                    u = otra_casilla[0]
                    v = otra_casilla[1]
                    if not formula_1:
                        formula_1 = self.Reina_En_Casilla.ravel([u, v])
                    else:
                        formula_1 = (
                            f"({formula_1}O{self.Reina_En_Casilla.ravel([u,v])})"
                        )
                formula_1 = f"({self.Reina_En_Casilla.ravel([col, fila])}>-{formula_1})"
                if not regla1:
                    regla1 = formula_1
                else:
                    regla1 = f"({regla1}Y{formula_1})"

        return regla1

    def regla2(self):
        """
        Regla 2: Por cada casilla del tablero, se tiene que cumplir que o bien no hay una reina en esa casilla o bien no hay una reina en la misma fila.

        La regla se escribe como: para cada casilla (x,y) del tablero, se tiene que ((no hay una reina en (x,y)) o (no hay una reina en la misma fila y)) para cada fila x diferente de x.

        Se implementa como: para cada casilla (x,y) del tablero, se construye una formula que es la disyuncin de (no hay una reina en (x,y)) y la negacion de la existencia de una reina en la fila y, para cada columna x diferente de x. Luego se construye la conjuncion de todas estas formulas.
        """
        regla2 = ""
        for col in range(self.board_size):
            for row in range(self.board_size):
                casilla = (col, row)
                otras_casillas = [
                    (i, row) for i in range(self.board_size) if (i, row) != casilla
                ]
                formula1 = ""
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
        """
        Regla 3: Por cada casilla del tablero, se tiene que cumplir que o bien no hay una reina en esa casilla o bien no hay una reina en la misma diagonal.

        La regla se escribe como: para cada casilla (x,y) del tablero, se tiene que ((no hay una reina en (x,y)) o (no hay una reina en la diagonal que pasa por (x,y)) para cada casilla que esta en la diagonal y diferente de (x,y).

        Se implementa como: para cada casilla (x,y) del tablero, se construye una formula que es la disyuncion de (no hay una reina en (x,y)) y la negacion de la existencia de una reina en la diagonal que pasa por (x,y) y diferente de (x,y). Luego se construye la conjuncion de todas estas formulas.
        """
        regla3 = ""
        for col in range(self.board_size):
            for row in range(self.board_size):
                casilla = (col, row)
                casillas_diagonal = [
                    (i, j)
                    for i in range(self.board_size)
                    for j in range(self.board_size)
                    if (i, j) != casilla and abs(col - i) == abs(row - j)
                ]
                formula1 = ""
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
        """
        Regla 4: Construye una regla para asegurar que haya exactamente una reina en cada fila
        del tablero. 

        Para cada combinación de posiciones de reinas en el tablero, se genera una fórmula 
        lógica que representa la conjunción de que hay una reina en cada una de esas posiciones.
        Luego, se construye una disyunción de todas estas fórmulas para definir la regla que 
        garantiza que haya una reina en cada fila del tablero.

        Returns
        -------
        str
            La regla en formato de lógica proposicional que asegura una reina por fila.
        """
        print("Building regla4...")
        n = self.board_size
        
        regla4 = ""
        for x in range(n):
            formula = ""
            for y in range(n):
                if not formula:
                    formula = self.Reina_En_Casilla.ravel([x, y])
                else:
                    formula = f"({formula}O{self.Reina_En_Casilla.ravel([x, y])})"
            
            if not regla4:
                regla4 = formula
            else:
                regla4 = f"({regla4}Y{formula})"

        return regla4
    
    def build_all_rules(self) -> str:
        """
        Construye la regla que indica que el tablero de ajedrez debe tener
        exactamente n reinas en n posiciones diferentes y no ataques entre ellas.

        Returns
        -------
        str
            La regla en formato de logica proposicional.

        """
        return reduce(lambda x, y: f"({x}Y{y})", self.reglas)
    
    def visualizar(self, I: dict[str, bool]) -> None:
        """
        Visualiza el tablero de ajedrez con las reinas en las posiciones
        especificadas en el diccionario I.

        Parameters
        ----------
        I : dict[str, bool]
            Diccionario donde las claves son las letras proposicionales del descriptor de las reinas 
            y los valores son booleanos que indican si la reina se encuentra
            en esa posición o no.

        Returns
        -------
        None
        """
        try:
            image = Image.open("img/queen.png")
        except FileNotFoundError:
            print("Error: The image path is incorrect or the file does not exist.")
            image = None

        if image:
            # Filter the letters that are in the range of the chessboard
            filtered_I = {}

            for letter in I:
                if (ord(letter) >= self.Reina_En_Casilla.rango[0]) and (ord(letter) <= self.Reina_En_Casilla.rango[1]):
                    filtered_I[letter] = I[letter]

            # Console print
            for letter in filtered_I:
                if filtered_I[letter]:
                    print(self.Reina_En_Casilla.escribir(letter))

            true_pos = [self.Reina_En_Casilla.unravel(letter) for letter in filtered_I if filtered_I[letter]]
            n = self.board_size

            # Create a checkerboard pattern for the chessboard
            chessboard = np.indices((n, n)).sum(axis=0) % 2
            colors = np.where(chessboard == 0, '#D9B58D', '#8B5A2B')  # Light blue and light gray colors

            # Set up the figure and axis
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw the chessboard
            for row in range(n):
                for col in range(n):
                    color = colors[row, col]
                    rect = plt.Rectangle((col, n - row - 1), 1, 1, facecolor=color, edgecolor='black')
                    ax.add_patch(rect)

            # Overlay the image on the true positions
            for pos in true_pos:
                col, row = pos
                img = OffsetImage(image, zoom=1)  # Adjust zoom for visibility
                ab = AnnotationBbox(img, (col + 0.5, n - row - 0.5), frameon=False)
                ax.add_artist(ab)

            # Set limits, ticks, and remove axes
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()  # Align the origin with the top left corner

            # Show plot
            plt.show()


def super_important_function_that_every_project_needs():
    print(
        """
  ／l、             
（ﾟ､ ｡ ７         
  l  ~ヽ       
  じしf_,)ノ        
"""
    )
