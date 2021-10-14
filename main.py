import math
import numpy as np
import random


def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if (h >= 0) and (r >= 0):
        return math.pi * r ** 2 * 2 + 2 * math.pi * r * h
    return float('nan')


def fib(n: int) -> np.array:
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if n == 1:
        return np.array([1])
    if n > 1 and isinstance(n, int):
        vector = np.array([1, 1])
        for x in range(2, n):
            vector = np.append(vector, vector[x - 2] + vector[x - 1])
        return np.array([vector])
    return None


def matrix_calculations(a: float) -> tuple:
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """

    m = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    mdet = np.linalg.det(m)
    if mdet == 0:
        minv = float('nan')
    else:
        minv = np.linalg.inv(m)
    mt = np.transpose(m)
    return minv, mt, mdet


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    data = (m, n)
    if (m <= 0 or n <= 0) or not (isinstance(m, int) and isinstance(n, int)):
        return None
    m = np.zeros(data)
    for x in range(0, data[0]):
        for y in range(0, data[1]):
            if x > y:
                m[x, y] = x
            else:
                m[x, y] = y
    return m


