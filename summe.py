# -*- coding: utf-8 -*-
#
# Summe aller positiven Einträge eines Vektors
#
#   s = summe_pos(x)
#
#   Eingabe
#       x       Vektor (numpy.ndarray mit einer Dimension)
#
#   Ausgabe
#       s       Summe aller positiven Einträge von x (Skalar)
#
import numpy as np

def summe_pos(x):
    # TODO: berechne s
    if isinstance(x, np.ndarray):
        s = np.where(x>0,x,0).sum()
    else:
        s = 0
    return s



vec = np.array([1,2,-3,4,-5,0])

print(summe_pos(vec))
