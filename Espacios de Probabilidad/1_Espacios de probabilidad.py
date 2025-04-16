#!/usr/bin/env python
# coding: utf-8

# # Universidad Nacional Autónoma de México
# # Facultad de Estudios Superiores Acatlán
# 
# # Diplomado en Técnicas Estadísticas y Minería de Datos
# 
# # Módulo II: Modelos Estadísticos
# 
# # Alejandra Belmont Valderrama

# In[65]:


import random 
import itertools #importamos el módulo
from itertools import product


# # Espacios de probabilidad

# ## Probabilidad Clásica
# Para un experimento aleatorio, se tiene que $\Omega$ es el espacio muestral.
# 
# Sea $A$ un evento, entonces
# 
# $$ \mathbb{P}(A) = \frac{\#A}{\#\Omega} $$
# 
# $\color{indigo}{\text{Ejemplo.}}$ Sea el experimento de lanzar una moneda 3 veces, de manera independiente (es decir, el $i$-ésimo lanzamiento no me afecta al $i+1$ lanzamiento). El espacio muestral de este experimento es:
# 
# $$ \Omega = \{A,S\} \times \{A,S\} \times \{A,S\} = \{A,S\}^3 $$
# y tenemos que $\#\Omega = 2^3$.
# 
# Sean los eventos:
# * $A$: todos los resultados son águila
# * $S$: todos los resultados son soles
# * $E$: Por lo menos un resultado es sol
#   
# Calcular las probabilidades de estos eventos:
# 
# $\color{indigo}{\text{Solución.}}$ Vamos a descomponer mi conjunto $A$ de la siguiente manera:
# $$A = A_1 \cap A_2 \cap A_3 $$
# donde $A_i$: caé águila en el $i$-ésimo resultado.

# In[57]:


# Vamos a crear el espacio muestral
import itertools #importamos el módulo
from itertools import product

Omega = set( product({"A","S"},repeat=3))
print(f"Omega: {Omega} \n")

Card = len (Omega) # Cardinalidad de Omega
print(f"Cardinalidad de omega: {Card} \n")

# Vamos a crear los eventos A, A_1, A_2 y A_3
A_1 = { om for om in Omega if om[0]=="A"} # La primera coordenada es águila
A_2 = { om for om in Omega if om[1]=="A"} # La segunda coordenada es águila
A_3 = { om for om in Omega if om[2]=="A"} # La tercera coordenada es águila
A= A_1.intersection(A_2.intersection(A_3))
print(f"Evento A. Todos los resultados son águila: {A} \n")
print(f"Evento A1. El primer resultado es águila: {A_1} \n")
print(f"Evento A2. El segundo resultado es águila: {A_2} \n")
print(f"Evento A3. El tercer resultado es águila: {A_3} \n")

#Vamos a calcular la probabilidad de A
P_A = len(A) / len(Omega)
print(f"Probabilidad de que todos los resultados sean águila: {P_A} \n")


# In[58]:


from fractions import Fraction
def p(E, Omega):
    p = Fraction(len(E),len(Omega))
    return p
prob_A = p(A,Omega)
prob_A


# $\color{purple}{\text{EJERCICIO.}}$ Dar solución a los eventos restantes.
# * $S$: Todos los resultados son sol

# In[59]:


# Vamos a crear los eventos S_1, S_2 y S_3
S_1 = { om for om in Omega if om[0]=="S"}
S_2 = { om for om in Omega if om[1]=="S"}
S_3 = { om for om in Omega if om[2]=="S"}
# Vamos a crear el evento S
S= S_1.intersection(S_2.intersection(S_3))

#Vamos a calcular la probabilidad de S
P_S = len(S) / len(Omega)
print(f"Probabilidad de que todos los resultados sean sol: {P_S} \n")


# * $E$: Por lo menos un resultado es sol

# In[60]:


#Vamos a calcular la probabilidad de E
P_E = 1-P_A
print(f"Probabilidad de que al menos un resultado sea sol: {P_E} \n")


# ## Probabilidad Condicional
# 
# Definimos la probabilidad de que ocurra un evento $A$ sabiendo que ocurre el evento $B$, 
# $$ \mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} = \frac{\# A\cap B}{\#B} $$
# donde $\mathbb{P}(B)>0$. 
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Quiero calcular la probabilidad condicional de los siguientes eventos:
# * $B$: El primer lanzamiento es águila
# * $C$: Dos de los resultados son águila

# In[61]:


# A_1 es el evento del primer lanzamiento es águila
B = { om for om in Omega if om[0] == "A"} # el primer lanzamiento es águila
C = { om for om in Omega if om.count("A") == 2}
def prob(E):
    return len(E) / len(Omega)

def prob_cond(E,F):
    return len(E&F)/len(F)
P_B = prob(B)
P_C = prob(C)
P_B_y_C = prob_cond(B,C)

print(f"Probabilidad de que el primer resultado sea águila: {P_B} \n")
print(f"Probabilidad de que dos de los resultados sean águila: {P_C} \n")
print(f"Probabilidad de que el primer resultado sea águila dado a que se tuvieron dos resultados águila: {P_B_y_C} \n")


# ## Independencia de Eventos
# 
# Decimos que dos eventos $A$ y $B$ son independientes si:
# $$ \mathbb{P}(A\cap B) = \mathbb{P}(A)\mathbb{P}(B) $$
# o de manera equivalente
# $$ \mathbb{P}(A|B)=\mathbb{P}(A) $$

# In[62]:


def indep(E,F):
    return prob(E&F)==prob(E)*prob(F)
Ind_B_y_C = indep(B,C)
print(f"¿Los eventos los que el primer resultado es águila y en el que se obtuvieron dos resultados águila son independientes? {Ind_B_y_C} \n")


# Vamos a considerar el evento:
# * $D$: en el segundo lanzamiento es sol

# In[63]:


D = {om for om in Omega if om[1]=="S"}
P_D = prob(D)
P_B_y_D = prob_cond(B,D)
Ind_B_y_D = indep(B,D)
print(f"Probabilidad de que el segundo resultado sea sol: {P_D} \n")
print(f"Probabilidad de que el segundo resultado sea sol dado a que el primer resultado sea águila: {P_B_y_D} \n")
print(f"¿Los eventos los que el segundo resultado es sol y en el que el primero sea águila son independientes?: {Ind_B_y_D} \n")


# Quiero verificar si $B$, $C$ Y $D$ son independientes:
# 
# Recordemos que
# * $B$: El primer lanzamiento es águila
# * $C$: Dos de los resultados son águila
# * $D$: en el segundo lanzamiento es sol

# In[64]:


Ind_B_y_D = indep(B,D)
Ind_C_y_D = indep(C,D)
Ind_B_y_C = indep(B,C)
print(f"Independencia entre B y D: {Ind_B_y_D} \n")
print(f"Independencia entre C y D: {Ind_C_y_D} \n")
print(f"Independencia entre B y C: {Ind_B_y_C} \n")


# Por lo tanto, los eventos no son independientes.

# $\color{indigo}{\text{EJEMPLO.}}$ Sea el experimento aleatorio de lanzar un dado dos veces. Sabemos que el espacio muestral es el siguiente:
#    $$ \Omega = \{(i,j) | i,j  \in \{1,2,3,4,5,6\}\} = \{1,2,3,4,5,6\}\times \{1,2,3,4,5,6\} = \{1,2,3,4,5,6\}^2$$
#    
#   y además $ \#\Omega = 6^2 = 36 $

# In[67]:


Omega_dado = set(product(["1","2","3","4","5","6"],repeat=2)) # Estamos haciendo los pares ordenados
print(f"Omega: {Omega_dado} \n")

Card = len (Omega_dado) # Cardinalidad de Omega
print(f"Cardinalidad de omega: {Card} \n")

def prob_dado(E):
    return len(E) / cardOm


# Definimos el evento:
# * $S_n$: la suma de los dados es $n$
# y escrito como conjunto es
# $$ S_n = \{ (i,j) \in\Omega\mid i+j =n \} $$

# In[69]:


def S(n):
    Sn = {(i,j) for i in range(1,7) for j in range(1,7) if i + j == n }
    return Sn
S_nueve = S(9)
print(f"Eventos en los que la suma de los tiros sea 9: {S_nueve} \n")

P_S_nueve = prob_dado(S_nueve)
print(f"Probabilidad de que la suma de los tiros sea 9: {S_nueve} \n")


# In[90]:


prob_S_nueve = p(S_nueve,Omega_dado)
prob_S_nueve


# $$ \mathbb{P}(S(9))=\frac{1}{9}$$

# ## Probabilidad Frecuentista
# 
# Tenemos la hipótesis de que el espacio muestral es un conjunto finito, y podemos definir la probabilidad asociada a un evento $A$:
# 
# $$ f_n(A) = \frac{n(A)}{n}, $$
# donde $n(A)$ denota al número de veces que ocurre $A$, y $n$ es el número de veces en que se realiza el experimento.

# $\color{indigo}{\text{EJEMPLO.}}$ Se tiene el lanzamiento de monedas injustas (volados)

# In[91]:


def volado(p):
    return 'sol' if random.random() < p else 'aguila'
volado(0.4)


# In[92]:


a = [volado(0.4) for i in range(10)]
a


# In[93]:


a.count("sol")


# In[94]:


resultados = [] # lista vacia
for i in range(10):
    resultado = volado(0.4)
    resultados.append(resultado)
resultados


# In[95]:


# Vamos a simular nuestra probabilidad frecuentista
p = [] #Lista para almacernar las probabilidades

for i in range(1000): #Simula 1000 experimentos
    N = 1000 # numero de lanzamientos 
    resultados = []

    for i in range(N):
        resultado = volado(0.63)
        resultados.append(resultado)

    n_soles = resultados.count("sol") # cuenta en número de soles
    p_soles = n_soles / N #calcula la proporción

    p.append(p_soles)

# Probabilidad promedio
p_soles_prom = sum(p) / 1000 

print("Probabiliad {:.3f}".format(p_soles_prom))   

