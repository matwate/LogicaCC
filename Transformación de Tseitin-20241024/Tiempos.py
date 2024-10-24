from time import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform, choice

def obtiene_tiempos(fun, args, num_it=100):
    tiempos_fun = []
    for i in range(num_it):
        arranca = time()
        x = fun(*args)
        para = time()
        tiempos_fun.append(para - arranca)
    return tiempos_fun

def compara_entradas_funs(funs, nombres_funs, lista_args, N=10):
    entradas = []
    funcion = []
    tiempos = []
    lista_dfs = []
    for i, args in enumerate(lista_args):
        for j, fun in enumerate(funs):
            t = obtiene_tiempos(fun, [args], N)
            tiempos += t
            n = len(t)
            entradas += [i+1]*n
            funcion += [nombres_funs[j]]*n
        df = pd.DataFrame({'Long_entrada':entradas,
                           'Funcion':funcion,
                           'Tiempo':tiempos})
        lista_dfs.append(df)
    df = pd.concat(lista_dfs).reset_index()
    sns.lineplot(x='Long_entrada',y='Tiempo',hue='Funcion',data=df)
    plt.show()

def lista_formulas(cantidad=20, tipo='aleatorio'):
    letras = [chr(i) for i in range(256, 256+cantidad)]
    lista = [letras[0]]
    formula = letras[0]
    if tipo == 'aleatorio':
        for p in letras[1:]:
            neg1 = '-' if uniform(0,1) > .5 else ''
            neg2 = '-' if uniform(0,1) > .5 else ''
            neg3 = '-' if uniform(0,1) > .5 else ''
            conectivo = choice(['Y','O','>'])
            if uniform(0,1) > .5:
                formula = neg1 + "(" + neg2 + formula + conectivo + neg3 + p + ")"
            else:
                formula = neg1 + "(" + neg2 + p + conectivo + neg3 + formula + ")"
            lista.append(formula)
    elif tipo == 'feo-fnc':
        for i in range(len(letras)-1):
            neg1 = '-' if uniform(0,1) > .5 else ''
            neg2 = '-' if uniform(0,1) > .5 else ''
            p = letras[i]
            q = letras[i+1]
            formula = "(" + formula + "Y(" + neg1 + p + 'Y' + neg2 + q + "))"
            lista.append(formula)
    else:
        raise Exception('tipo desconocido!')
    return lista
