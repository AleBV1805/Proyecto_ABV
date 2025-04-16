#!/usr/bin/env python
# coding: utf-8

# # Variables Aleatorias

# In[3]:


# Librerías

# Todos
import numpy as np # Importar la libreria

# Variables Aleatorias
#usamos itertools
from itertools import product
from fractions import Fraction

# Importamos la libreria Pandas
import pandas as pd

#Variables Aleatorias Discretas
import numpy.random as npr # Para generar números aleatorios
import scipy.stats as sps # Paqueterías estadísticas
from scipy.stats import randint #Para trabajar una uniforme discreta en un rango de enteros

# Para visualizar
import matplotlib.pyplot as plt # Para visualizar datos

import random #para generar números aleatorios

#Para la tarea de V.A. Discretas
from scipy.stats import poisson
from scipy.stats import hypergeom
from math import comb

# V.A. Continuas

import numpy as np # Importamos Numpy
import numpy.random as npr # para generar numeros aleatorios

import scipy.stats as sps # Importamos el modulo SciPy
from scipy.stats import randint # para trabajar una uniforme discreta en un rango de enteros

import matplotlib.pyplot as plt #Visualizar datos 

import random # para generar numeros aleatorios


# ## El espacio muestral asociado a un experimento aleatorio
# 
# El espacio muestral $\Omega$ del experimento aleatorio de lanzar dos dados honestos, esta compuesto por todas las pearejas ordenadas $(i,j)$ tales que $1 \leq i,j \leq 6$
# 
# $$\Omega = \{ (i,j) \mid 1 \leq i,j \leq 6 \} $$
# y sabemos que $\#\Omega = 6^2 = 36$

# In[4]:


Omega = set(product([1,2,3,4,5,6], repeat = 2))
card_Omega = len(Omega)

print(f"El espacio muestral es: {Omega} \n")
print(f"La cardinalidad es: {card_Omega} \n")


# Sea $S_n$ el evento "La suma de los dados en $n$". Esto es: 
# $$S_n=\{(i,j)\in\Omega:i+j=n\}.$$

# In[5]:


def S(n):
    Sn = {(i,j) for i in range(1,7) for j in range(1,7) if i+j==n}
    return Sn
print(f"El evento 𝑆9 es: {S(9)} \n")
print(f"El evento 𝑆12 es: {S(12)} \n")


# ## La variable aleatoria asociada a $S_n$

# In[6]:


# Definir la funcion P(A)
# Es la probabilidad del evento A definida mediante la definición clásica

def P(A):
    P = Fraction(len(A), len(Omega))
    return P

# Vamos a definir un diccionario S
# (i,j) pares ordenados y los valores de su suma

S = {(i,j) : i+j for i,j in Omega} 

# Agrupación de combinaciones por suma

from collections import defaultdict

dS = defaultdict(set)

# Vamos a recorrer sobre los elementos en S

for i,j in S.items():
    
    dS[j].add(i)
    
list(dS.items()) 


# In[7]:


# LeyS almacenar la probabilidad de cada posible suma,
# Usando la función P(A)
# En otras palabras "la función de densidad"
# A es el conjunto de pares que producden la suma i

leyS = {i : P(A) for i, A in dS.items() }
leyS


# Recordemos que una variable aleatoria real $X$, es una función
# $$X : \Omega \to \mathbb{R}$$
# 
# Sabemos que la función de densidad de una v.a. discreta $X$, se define como sigue:
# $$ f_X(x) = \begin{cases} \mathbb{P}(X=x), \quad \text{si } x\in R_{X} \\ 0, \quad \text{e.o.c} \end{cases} $$
# donde $R_X$ es el rango de la v.a. $X$.
# 
# Para nuestro evento $S_n$, se tiene la siguiente variable aleatoria
# $$ S : \Omega \to \{2,3,4,5,6,7,8,9,10,11,12\}$$

# In[8]:


# Vamos a crear una serie 

ley_S = pd.Series(leyS)
ley_S.sort_index() #ordenar


# In[9]:


# LeyS(i) devuelve la probabilidad de obtener la suma i, si está en el rango de posibles sumas

def leyS(i):
    if i in range(2,13):
        x = ley_S[i]
    else:
        x = 0 # si no está en el rango me devuelve cero
    return x

# Vamos a crear el rango de S

#creo una lista con los posibles valores
rango_S = [k for k in range(2,13)]

#Obtenemos la probabilidad de cada suma 
p_k = [float(leyS(k)) for k in rango_S]

# Creamos un DataFrame con:
# - los valores del rango_S
# - las probabilidades p_k
# Indexamos una etiqueta S_2,...,S_12

# zip() -> toma dos listas y las combina en pares ordenados
lS = pd.DataFrame(list(zip(rango_S, p_k)), index=[f"S_{i}" for i in range(2,13)], columns = ['rango','densidad'])
lS


# In[10]:


lS.plot.bar( x= 'rango', y = 'densidad')


# ## Variables Aleatorias Discretas
# 
# ### Variable aleatoria uniforme discreta
# 
# Una variable aleatoria $X$ tiene distribución uniforme discreta en el conjunto $\{x_1,...,x_n\}$ si su función de densidad está dada por
# $$ f_X(x) = \mathbb{P}(X=x) = \begin{cases} \frac{1}{n},\quad \text{si } x\in \{x_1,...,x_n\} \\
# 0, \quad \text{en otro caso} \end{cases}$$
# 
# Se llama uniforme porque cada uno de sus posibles resultados de $X$ tiene la misma probabilidad.
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim Unif(x_1,...,x_n)$
# 
# La esperanza y varianza de $X$, están dadas por:
# $$\mathbb{E}[X]= \sum_{x} xf_X(x) = \frac{x_n + x_1}{2}$$
# 
# $$ \text{Var}(X) = \mathbb{E}\left[ (X-\mathbb{E}[X])^2\right] = \frac{(x_n - x_1 + 1)^2 -1}{12}$$
# 
# $\color{purple}{\text{EJERCICIO 1.}}$ Verificar que, en efecto, la esperanza y varianza de una variable aleatoria es uniforme.
# 

# Supongamos que $\mathbb{X}$ es una v.a. discreta con distribución uniforme en el conjunto $\{x_1,...,x_n\} \in \mathbb{Z}_{+}$ y ${x_1 \leq x_n}$.
# 
# Entonces, la esperanza y varianza de $X$ están dados por:
# \begin{align*}
#            E[X] & = \sum_{k=x_1}^{x_n}k\left(\mathbb{P}{(X=k)}\right) \\
#                 & = \frac{1}{n} \sum_{k=x_1}^{x_n}k\\
#                 & = \frac{1}{n} \left(\frac{n}{2} (x_n+x_1)\right)\\
#                 & = \frac{x_n+x_1}{2} \\
# \end{align*}

# La varianza se define como:
# $$\text{Var}(X) = E[X^2] - (E[X])^2$$
# 
# Por lo que es necesario calcular $E[X^2]$
# 
# \begin{align*}
#               E[X^2] & = \sum_{k=x_1}^{x_n} k^2 P(X = k) \\
#                               & = \frac{1}{x_n-x_1+1} \sum_{k=x_1}^{x_n} k^2 \\
#                               & = \frac{(x_n+x_1)(2x_n+2x_1+1)}{6} \\
# \end{align*}
# 
# De manera que:
# \begin{align*}
#               \text{Var}(X) & = \frac{(x_n+x_1)(2x_n+2x_1+1)}{6} - \left(\frac{x_1+x_n}{2}\right)^2 \\
#                             & = \frac{(x_n+x_1)(2x_n+2x_1+1)}{6} - \frac{(x_1+x_n)^2}{4} \\
#                             & = \frac{(x_n-x_1+1)^2 - 1}{12}
# \end{align*}
# 

# La función de distribución de una v.a. uniforme es:
# $$ F_X(x) = \mathbb{P}(X\le x) = \begin{cases} 0, \quad \text{si } x < x_1 \\ \frac{x}{n}, \quad \text{si } x\in \{x_1,...,x_n\} \\
# 1, \quad \text{si } x > x_n \end{cases}$$
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Se lleva a cabo una rifa donde los boletos están enumerados del $00$ al $99$. Si $Y$ es la variable aleatoria definida como el número del boleto ganador, entonces:  
# $$\mathbb{P}(Y=k)=\begin{cases}
# \frac{1}{100} &\quad \text{si } k=00,01,\dots,99 \\
# 	0 &\quad \text{eoc}
# \end{cases}$$
# Consideremos que el premio de la rifa se determina a partir del número premiado de la siguiente forma: $X=Y+1$, donde $X$ es el monto del premio en pesos y $Y$ es el número premiado, entonces $X$ es una variable aleatoria, pues es una función de $Y$, y además se tiene
# 	$$f_{X}(k)=\begin{cases}
# 		\frac{1}{100} &\quad \text{si } x=1,2,\dots,100 \\
# 		0 &\quad \text{eoc}
# 	\end{cases}$$
#     
# $\color{purple}{\text{EJERCICIO 2.}}$ Calcular la esperanza y varianza. 
# 
#   $$E(X) = \sum_{k=00}^{99} X \frac {1}{100}$$

# In[11]:


xn=100
x1=1
esperanza=(xn+x1)/2
varianza= (((xn-x1+1)^2)-1)/12
print('La esperanza es =',esperanza)
print('La varianza es =',varianza)


# Entonces, la esperanza y varianza de $X$ están dados por:
# $$E[X] = \left[\frac{(100+1)}{2}\right] = 50.5$$
# 	y
# $$\text{Var}(X) = \frac{(100-1+1)^2 - 1}{12} = 8.42 $$
# 
# 
#     
# Supongamos que nos interesa calcular la probabilidad de que el premio sea mayor a $\$80$, entonces
# 	$$\mathbb{P}(X>80) = \sum_{k=81}^{100}\frac{1}{100} = \frac{20}{100} = 0.2$$
# 	Su función de distribución esta dada por:
# 	$$F_{X}(x)=\mathbb{P}(X\le x)=\begin{cases}
# 		0 &\quad \text{si } x<1 \\
# 		\frac{x}{100} &\quad \text{si } x=1,2,\dots,100\\ 1 &\quad \text{si } x>100
# 	\end{cases}$$

# ### Gráfica de la distribución Uniforme Discreta

# In[12]:


# Definir el rango de la distribución uniforme discreta [low,high)
low = 0 # límite inferior 
high = 20 # límite superior

# Vamos a crear los valores posibles dentro del rango, mediante un arrange
x = np.arange(low,high)

# Calcular la función de masa de probabilidad (o función de densidad) de 
# una distribución uniforme discreta.
# (Probability mass function)
# Va a asignar una probabilidad entre cada valor entre el límite inferior 
# y superior de manera que la suma de las probabilidades sea 1
pmf = np.full_like(x,1/(high-low), dtype = float)

# Para crear una figura más grande
#plt.figure(figsize=(12,6))

# Graficar pmf
plt.bar(x,pmf, width=0.2 ,color='purple',edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Probabilidades')
plt.title('Función de densidad de una uniforme discreta')
#plt.grid(True)
plt.show() # Para mostrar el gráfico

# https://matplotlib.org/stable/gallery/color/named_colors.html


# ### Variable aleatoria Bernoulli con parámetro $p\in (0,1)$.

# El modelo probabilístico Bernoulli se aplica a un experimento cuyo espacio muestral está constituido sólo por dos resultados posibles, éxito y fracaso:
# 
# Se considerará una v.a $X$ sobre el espacio muestral
# $$ \Omega = \{\text{exito} , \text{fracaso}\} $$
# de tal forma que
# - $X(\{\text{exito}\}) =1 $
# - $ X(\{\text{fracaso}\}) =0 $
# 
# Las probabilidades asociadas a este modelo son:
# - $ \mathbb{P}(\{\text{éxito}\}) = p $
# - $ \mathbb{P}(\{\text{fracaso}\}) = 1-p $
# 
# donde $0<p<1$.
# 
# La función de densidad, está definida de la siguiente manera:
# $$ f_X(x) = \mathbb{P}(X=x) = \begin{cases}p^x(1-p)^{1-x}, \quad \text{si } x\in \{0,1\} \\  0, \quad \text{e.o.c.} \end{cases}$$
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim Ber(p)$
# 
# Esta variable aleatoria es muy importante. En la practica es usada para modelar situaciones en las cuales hay dos posibles resultados como:
# * el estado de un teléfono en un momento dado: libre u ocupado.
# * una persona pueda estar enferma o sana de una determinada enfermedad.
# * la preferencia de una persona, la cual puede estar a favor o en contra de un candidato determinado.
# 
# Mediante la combinación de v.a. Bernoulli es posible construir otras v.a.
# 
# #### Esperanza y Varianza
# 
# La esperanza y varianza de $X\sim Ber(p)$, está dada por:
# $$ E[X] = p $$
# y
# $$ \text{Var}(X) = p(1-p)$$
# 
# Para más información, ver [Bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)

# ### Gráfica de la distribución Bernoulli

# In[13]:


# Importamos Bernoulli
from scipy.stats import bernoulli

# Definimos nuestra probabilidad de éxito
p=0.3

# Definiendo los posibles valores
x=[0,1]

# Calcular la función de masa de probabilidad (o función de densidad).
# (Probability mass function)
# Va a asignar una probabilidad entre cada valor entre el límite inferior 
# y superior de manera que la suma de las probabilidades sea 1
pmf = bernoulli.pmf(x,p)

print("La función de densidad = ", pmf)

# Graficamos la función de densidad
#plt.figure(figsize=(12,6))
plt.bar(x,pmf, width=0.2 ,color='pink',edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Probabilidades')
plt.title('Función de densidad de una bernoulli con p=0.3')
#plt.grid(True)
plt.show() # Para mostrar el gráfico

# https://matplotlib.org/stable/gallery/color/named_colors.html

# Cálculo de la esperanza
print("La esperanza = ",bernoulli.mean(p))

# Cálculo de la varianza
print("La varianza = ",bernoulli.var(p))


# ### Variable aleatoria Binomial con parámetros $ n $ y $p \in (0,1)$

# Decimos que una v.a. $X$ tiene distribución binomial con parámetros $n$ y $p$, si su función de densidad está dada por:
# $$ f_X(x) = \mathbb{P}(X=x) = \begin{cases} \binom{n}{x} p^x (1-p)^{n-x}, \quad \text{si } x\in \{0,1,...n\} \\ 0, \quad \text{e.o.c.} \end{cases} $$
# donde $ n\in \mathbb{Z}_{+}$ y ${0<p<1}$.
# 
# $\color{slateblue}{\text{OBSERVACIÓN.}}$ Los ensayos deben ser independientes.
# 
# La esperanza de $X$ es:
# $$ E[X] = np $$
# 
# y la varianza de $X$ es:
# $$\text{Var}(X)=np(1-p).$$
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Una moneda justa se tira seis veces, donde la probabilidad de obtener sol es de $0.3$. Sea $X$ el número de veces que cael sol, entonces 
# sabemos que $X$ tiene una distribución binomial con parámetros $n=6$ y $p=0.3$. Calcular:
# - $ \mathbb {P}(X=2) = \binom{6}{2}(0.3)^2 (1-p)^{6-2} = 0.3241 $
# - $ \mathbb {P}(X=3) = \binom{6}{3}(0.3)^3 (1-p)^{6-3} = 0.1852 $
# - $$ \begin{align*} \mathbb{P}(1<X \le 5) &= \mathbb{P}(X=2) + \mathbb{P}(X=3) + \mathbb{P}(X=4) + \mathbb{P}(X=5) \\
#     &=0.3241 + 0.1852 + \binom{6}{4}(0.3)^4 (1-p)^{6-4} + \binom{6}{5}(0.3)^5 (1-p)^{6-5} \\
#     &=0.579 \end{align*}$$
#     
#  Para más información, ver [Binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
# 
# Notemos que $\mathbb{P}(1<X\le 5) = F_X(5)-F_X(1)$
# 
# Usamos el atributo `.cdf` para calcular estas probabilidades

# In[14]:


from scipy.stats import binom
a = binom.cdf(5,6,0.3) # Calcula la probabilidad acumulada de que haya 5 o menos éxitos en 6 ensayos
b = binom.cdf(1,6,0.3) # Calcula la probabilidad acumulada de que haya 1 o menos éxitos en 6 ensayos
a-b


# ### Gráfica de la distribución Binomial

# In[15]:


# Número de ensallos bernoulli
n = 25 

# Probabilidad de éxito
p=0.3

# Número de muestras
s= 1000

# Vamos a generar números aleatorios que siguen una distribución binomial
binom_numeros = sps.binom.rvs(n, p, size = s)

# Creamos histograma
plt.figure(figsize=(10,6))
plt.hist(
    binom_numeros,
    density=True, #normaliza el área para que sea 1
    bins = len(np.unique(binom_numeros)), # número de barras del histograma
    color = "indigo",
    edgecolor = "grey"
    )
plt.xlabel('Valores')
plt.ylabel('Probabilidades')
plt.title('Función de densidad de una binomial')
plt.show() # Para mostrar el gráfico


# ### Variable aleatoria Poisson
# 
# Es una distribución de probabilidad discreta que sirve para calcular la probabilidad de que ocuirra un determinado número de eventos raros durante un intervalo dado (puede ser tiempo, longitud, área, etc.).
# 
# Esta v.a. toma valores sobre el conjjunto $ \{0,1,2...\} $ y tiene un parámetro $\lambda>0$, el cual representa el número de veces que se **espera** que ocurra un evento durante un intervalo dado.
# 
# Su función de densidad, está dado como sigue:
# 
# $$ f_X(x) = \mathbb{P}(X=x) = \begin{cases} e^{-\lambda}\frac{\lambda^x}{x!}, \quad \text{si } x\in \{0,1,...\} \\
# 0, \quad \text{e.o.c.} \end{cases}$$
# 
# En efecto, es una función de masa de probabilidades debido a que
# $$\begin{align*}
# \sum_{k=0}^{\infty}e^{-\lambda}\frac{\lambda^{k}}{k!} &= e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^{k}}{k!} \\ &= e^{-\lambda}\underbrace{\left(1+\lambda+\frac{\lambda^{2}}{2!}+\frac{\lambda^{3}}{3!}+\cdots\right)}_{\text{desarrollo de Taylor de }e^{\lambda}} \\ &= e^{-\lambda}e^{\lambda}=1.
# \end{align*}$$
# 
# Notemos que $E[X]=\lambda$ y que $\text{Var}(X)=\lambda$.
# 

# $\color{indigo}{\text{EJEMPLO.}}$ Supongamos que el número de accidentes que ocurre en un punto en un día tiene distribución Poisson con parámetro $\lambda=2$, 
# - ¿Cuál es la probabilidad de que en un día ocurran más de dos accidentes?
# $$ \begin{align*} \mathbb{P}(X>2) & = 1-\mathbb{P}(X\le 2) \\
#     &= 1-[\mathbb{P}(X=0) + \mathbb{P}(X=1) + \mathbb{P}(X=2)] \\
#     &= 1-\left[e^{-2}\frac{2^0}{0!} + e^{-2}\frac{2^1}{1!} + e^{-2}\frac{2^2}{2!} \right] \\
#     &= 1 - e^{-2}[1+2+2] = 1-5e^{-2} = 0.3233
#     \end{align*}$$
#     
# - ¿Cuál es la probabilidad de que ocurran más de dos accidentes sabiendo que por lo menos ocurre uno?
# $$\begin{align*}
# \mathbb{P}(X>2 \mid X\ge 1) = \frac{\mathbb{P}(X>2 \cap X\ge 1)}{\mathbb{P}(X\ge 1)} = \frac{\mathbb{P}(X>2)}{\mathbb{P}(X\ge 1)} = \frac{1-\mathbb{P}(X\leq 2)}{1-\mathbb{P}(X<1)} = \frac{1-5e^{-2}}{1-e^{-2}} = \frac{0.3233}{0.8646} = 0.3739
# \end{align*}$$
# ya que $\mathbb{P}(X\ge 1)=1-\mathbb{P}(X<1)=1-\mathbb{P}(X=0)=1-e^{-2}$

# $\color{purple}{\text{EJERCICIO 3.}}$ Usando el atributo `.cdf` [Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) calcula las probabilidades anteriores.
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Supongamos que el número de accidentes que ocurre en un punto en un día tiene distribución Poisson con parámetro $\lambda=2$.  
# - ¿cuál es la probabilidad de que en un día ocurran más de dos accidentes?
# - ¿cuál es la probabilidad de que ocurran más de dos accidentes sabiendo que por lo menos ocurre uno?

# In[16]:


lambda_p=2

#Probabilidad X>2
proba1 = 1 - poisson.cdf(2, lambda_p)

#Probabilidad de que ocurre por lo menos un accidente
proba2 = 1 - poisson.cdf(0, lambda_p)

proba3=proba1/proba2

print ('La probabilidad de que ocurran mas de dos accidentes es:', proba1)
print ('La probabilidad de que ocurran dos accidentes sabiendo que por lo menos ocurre uno', proba3)


# ### Aproximación de Poisson a la Binomial
# La distribución de Poisson es una forma límite de la distribución binomial, es decir, es una buena aproximación cuando $n$ es suficientemente grande y $p$ suficientemente pequeña.
# 
# $\color{indigo}{\text{Teorema (Poisson).-}}$ Sean $S_{n}\sim Bin(n,p_{n})$ bajo el regimen $$\lim_{n\to \infty}np_{n}=\lambda>0.$$
# Consideremos la siguiente sucesión de números reales:
# $$a_{j}(n,p_n)=\begin{cases}\binom{n}{j}(p_n)^{j}(1-p_{n})^{n-j} & j\leq n\\
# 0 & j\geq n+1\end{cases}$$
# 
#  Entonces,
#  $$\lim_{n\to \infty}a_{j}(n,p_n)=a_{j}=e^{-\lambda}\frac{\lambda^{j}}{j!} \ \ \forall j\in \mathbb{N}.$$
# 
#  El teorema anterior implica que la distribución de Poisson ofrece un modelo probabilístico adecuado para todos aquellos experimentos aleatorios 	en los que las repeticiones son independientes unas de otras y en los 	que sólo hay dos posibles resultados: éxito o fracaso, con probabilidad de 	éxito pequeña, y en los que el interés se centra en conocer el número de éxitos obtenidos al realizar el experimento un número suficientemente grande de veces.
# 
# Empíricamente se ha establecido, que la aproximación se puede aplicar con seguridad si $n\ge100$, $p\le 0.01$ y $np \le20$.
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Supongamos que la probabilidad de que un producto producido por cierta máquina es defectuoso es de $0.1$. ¿Cuál es la probabilidad de que un lote de 10 productos contenga a lo más un producto defectuoso?
# 
# Sea $X$ el número de productos defectuosos, y sabemos que $X$ tiene una distribución binomial con parámetros $n=10$ y $p=0.1$, entonces
# \begin{align*}
# \mathbb{P}(X\le 1) &= \mathbb{P}(X=0)+\mathbb{P}(X=1) \\ &= \binom{10}{0}(0.1)^{0}(0.9)^{10-0}+\binom{10}{1}(0.1)^{1}(0.9)^{10-1} \\ &= 0.7361
# 	\end{align*}
# 
# Ahora, con la distribución Poisson, tenemos que $\lambda=10(0.1)=1$, por lo que
# $$\mathbb{P}(X\le 1) = \mathbb{P}(X=0)+\mathbb{P}(X=1) = \frac{e^{-1}1^{0}}{0!}+\frac{e^{-1}1^{1}}{1!} = e^{-1}+e^{-1} =0.7358$$

# ### Gráfica de la aproximación de la Binomial a la Poisson

# In[17]:


# Simulación de la aproximación de la Bonomial a la Poisson
param=3 # Parametro de la Poisson que queremos aproximar
n=1000 # Este es el número de ensayos en la distribución binomial
N=5000 # Este es el número de simulaciones que realizaremos.

# Genera una muestra de N valores aleatorios de una distribución binomial con parámetros:
# n = 1000 (número de ensayos), p = param/n = 3/1000
X=npr.binomial(n,param/n,N)

# Calcular la frecuencia relativa de los valores simulados
counts = np.bincount(X) / float(N)

# Crear un array de valores posibles
x = np.arange(len(counts))

# Calcular la función de masa de probabilidad (FMP) de la distribución Poisson teórica
f_x = sps.poisson.pmf(x, param)

plt.close()
plt.bar(x - 0.5, counts, color = "slateblue", width=1., label="ley empírica")
p2 = plt.stem(x, f_x, "firebrick", label="ley teórica")
plt.legend()
plt.show()


# ### Variable aleatoria Geométrica con parámetro $p\in (0,1)$.
# 
# Esta variable aleatoria cuenta el número de fracasos antes del primer éxito en ensayos bernoulli independientes con parámetro $0<p<1$, y su función de masa de probabilidades está dada por:
# $$f_{X}(k)=\begin{cases}
# 	p(1-p)^{k-1} &\quad \text{si } k=1,2,\dots \\
# 	\qquad 0 &\quad \text{en otro caso}  
# \end{cases}$$
# 
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim Geo(p)$
# 
# La cual es una función de densidad ya que:
# * $$0\le f_{X}(k)\le 1 \forall x$$
# * $$\begin{align*}
# 	\sum_{k\in R_{X}}f_{X}(k) &= \sum_{k=1}^{\infty}(1-p)^{k-1}p \\
# 			&= p\sum_{y=0}^{\infty}(1-p)^{y} \\
# 			&= p\left(\frac{1}{1-(1-p)}\right) = 1
# 		\end{align*}$$
# 
# Si por el contrario queremos contar el número de éxitos antes del primer fracaso, tenemos que la función de está dada por:
# 	$$f_{X}(k)=\begin{cases}
# 		p^{k}(1-p) &\quad \text{si } k=0,1,2,\dots \\
# 		\qquad 0 &\quad \text{en otro caso}  
# 	\end{cases}$$
# 
# La esperanza de $X$ es:
# $$ E[X] = \frac{1}{p}$$
# 
# La varianza de $X$ es:
# $$\text{Var}(X)=\frac{1-p}{p^2}.$$

# ### Gráfica de la distribución geométrica

# In[18]:


p = 0.6 # probabilidad de éxito
s= 100000 # número de muestras

random.seed(3) #fijar una semilla
#Vamos a generar numeros aleatorios que siguen una distribución geométrica
geom_numeros = sps.geom.rvs(p,size=s) 

#Creamos un histograma
plt.figure(figsize=(10,6))
plt.hist(
    geom_numeros,
    density=True, # Normaliza el area para que sea 1
    bins=len(np.unique(geom_numeros)), # número de barras del histograma
    color = "darkcyan",
    edgecolor="azure" 
)

plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.title('Función de densidad de una geométrica')
plt.show() #mostrar el gráfica


# ### Variable aleatoria Binomial Negativa con parámetros $r\geq 1$ y $p\in (0,1)$.
# 
# Supongamos que se realizan ensayos independientes, cada uno con probabilidad $0<p<1$ de ser un éxito, hasta obtener un total de $r$ éxitos acumulados. Sea $X$ el número de ensayos que se requieren, entonces su función de masa de probabilidades está dada por:
# $$f_{X}(k)=\begin{cases}
# 	\binom{n-1}{r-1}p^{r}(1-p)^{n-r} &\quad \text{si } n=r,r+1,\dots \\
# 	\qquad 0 &\quad \text{en otro caso}  
# \end{cases}$$
# 
# 
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim \text{BN}(r,p)$.
# 
# Se tiene que
# $$E[X]=\frac{r}{p}$$
# y
# $$\text{Var}[X]=\frac{r(1-p)}{p^2}.$$

# ### Gráfica de la distribución Binomial Negativa

# In[19]:


r = 10  
p = 0.4 # probabilidad de éxito
s= 100000 # número de muestras

random.seed(3) #fijar una semilla
#Vamos a generar numeros aleatorios que siguen una distribución geométrica
nbinom_numeros = sps.nbinom.rvs(r,p,size=s) 

#Creamos un histograma
plt.figure(figsize=(10,6))
plt.hist(
    nbinom_numeros,
    density=True, # Normaliza el area para que sea 1
    bins=len(np.unique(nbinom_numeros)), # número de barras del histograma
    color = "olive",
    edgecolor="khaki" 
)

plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.title('Función de densidad de una BN con r=10 y p=0.4')
plt.show() #mostrar el gráfica


# $\color{purple}{\text{EJERCICIO 4.}}$ Un examen de Estadística consta de 20 preguntas tipo test y se conoce de experiencias
# anteriores que un alumno tiene probabilidad 0.7 de contestar bien cada pregunta. Obtener:
# 
# a) La probabilidad de que la primera pregunta que contesta bien sea la cuarta.
# 
# b) Sabiendo que para aprobar el examen es necesario contestar bien a 10 preguntas, ¿cuál es la probabilidad de que apruebe al contestar la pregunta duodécima?

# In[20]:


# Importamos las librerías necesarias
import numpy as np
import scipy.stats as sps  # Paqueterías estadísticas
import matplotlib.pyplot as plt # Para visualizar datos


p = 0.7  # Probabilidad de contestar bien una pregunta

# a) Probabilidad de que la primera pregunta que contesta bien sea la cuarta.

prob_a = p * (1 - p) ** (4 - 1)  # Fórmula de la distribución geométrica

print(f"a) La probabilidad de que la primera pregunta correcta sea la cuarta es: {prob_a:.4f}")

# b) Probabilidad de que apruebe al contestar la pregunta duodécima.

r = 10  # Número de preguntas correctas necesarias para aprobar
s = 100000  # Número de muestras
prob_b = sps.nbinom.pmf(12 - r, r, p)  # Fórmula de la binomial negativa
print(f"b) La probabilidad de aprobar al contestar la pregunta duodécima es: {prob_b:.4f}")

# Generar números aleatorios que siguen una distribución binomial negativa
nbinom_numeros = sps.nbinom.rvs(r, p, size=s)

# Crear un histograma
plt.figure(figsize=(10, 6))
plt.hist(
    nbinom_numeros,
    density=True,  # Normaliza el área para que sea 1
    bins=len(np.unique(nbinom_numeros)),  # Número de barras del histograma
    color="olive",
    edgecolor="khaki"
)

plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.title('Función de densidad de una Binomial Negativa con r=10 y p=0.7')
plt.show()


# $\color{purple}{\text{EJERCICIO 5.}}$ Investigue sobre el problema de la caja de cerillos de Banach y explique su solución.

# El problema de la caja de cerillos de Banach es un interesante concepto en la teoría de conjuntos y la topología. Se presenta en el contexto de la teoría de la medida y se refiere a la posibilidad de cubrir un conjunto no medible con una colección de conjuntos medibles.
# 
# En términos simples, imagina que tienes una caja de cerillos (o fósforos) y quieres saber si puedes organizar esos cerillos de tal manera que puedas cubrir un conjunto que no tiene un "tamaño" bien definido (es decir, que no se puede medir de manera convencional). El problema se centra en la existencia de conjuntos que son "raros" en el sentido de que no se pueden medir con la noción estándar de longitud, área o volumen.
# 
# La solución a este problema se relaciona con la construcción de conjuntos no medibles, como el conjunto de Vitali, que se forma utilizando la relación de equivalencia en los números reales. Este conjunto se puede construir eligiendo un representante de cada clase de equivalencia de números reales bajo la relación de ser "diferentes por un número racional". A través de este proceso, se demuestra que es posible crear un conjunto que no puede ser medido en el sentido tradicional.

# ### Variable aleatoria Hipergeométrica con parámetros $n,N,m$.
# 
# Supongamos que se elige, sin reemplazo, una muestra de tamaño $n$ de una urna que contiene $N$ bolas, de las cuales $m$ son rojas y $N-m$ son verdes. Sea $X$ el número de ebolas rojas seleccionadas, entonces su función de masa de probabilidades está dada por:
# $$f_{X}(k)=
# 	\frac{\binom{m}{i}\binom{N-m}{n-i}}{\binom{N}{n}} \quad \text{si } i=0,1,\dots, n  
# $$
# 
# 
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim \text{Hiper}(n,N,m)$.
# 
# Se tiene que
# $$E[X]=\frac{nm}{N}$$
# y
# $$\text{Var}[X]=\frac{nm}{N}\left[\frac{(n-1)(m-1)}{N-1}+1-\frac{nm}{N} \right].$$
# 
# $\color{slateblue}{\text{NOTA.}}$ Si $i\leq n$ y $X\sim \text{Hiper}(n,N,m)$, cuando $p=\frac{m}{N}$ y $m,N$ son muy grandes con respecto a $n$ e $i$:
# $$\mathbb{P}(X=i)\approx \binom{n}{i}p^{i}(1-p)^{n-i}.$$

# $\color{purple}{\text{EJERCICIO 6.}}$ Replica la gráfica de la funcion de densidad de una distribución hipergeometrica.

# In[21]:


def plot_hipergeometrica(N, m, n):
    """
    Grafica la función de masa de probabilidad de la distribución Hipergeométrica.

    Parámetros:
    N: Tamaño total de la población
    m: Número total de éxitos en la población
    n: Tamaño de la muestra
    """
    x = np.arange(0, min(m, n) + 1)  # Valores posibles de la variable aleatoria
    pmf = hypergeom.pmf(x, N, m, n)  # Función de masa de probabilidad

    plt.figure(figsize=(8, 5))
    plt.stem(x, pmf, basefmt="b")  # Gráfica en forma de línea con puntos
    plt.xlabel('Número de éxitos en la muestra')
    plt.ylabel('Probabilidad')
    plt.title(f'Distribución Hipergeométrica (N={N}, m={m}, n={n})')
    plt.grid()
    plt.show()
    
plot_hipergeometrica(N=50, m=20, n=10)


# $\color{purple}{\text{EJERCICIO 7.}}$ Una compañía petrolera realiza un estudio geológico que indica que un pozo petrolero exploratorio debería tener un 20% de posibilidades de encontrar petróleo.
# 
# - ¿Cuál es la probabilidad de que el primer pozo se produzca en el tercer pozo perforado?
# 
# - ¿Cuál es la probabilidad de que el tercer pozo se produzca en el séptimo pozo perforado?
# 
# - ¿Cuál es la media y la varianza del número de pozos que se deben perforar si la compañía petrolera quiere establecer tres pozos productores?

# In[22]:


# Probabilidad de encontrar petróleo
p = 0.2
q = 1 - p  # Probabilidad de no encontrar petróleo

# 1. Probabilidad de que el primer pozo se produzca en el tercer pozo perforado
# Fórmula: P(X = k) = q^(k-1) * p (Distribución geométrica)
k1 = 3
probabilidad_1 = q**(k1 - 1) * p
print(f"Probabilidad de que el primer pozo se produzca en el tercer pozo perforado: {probabilidad_1:.4f}")

# 2. Probabilidad de que el tercer pozo se produzca en el séptimo pozo perforado
# Fórmula: P(X = x) = comb(x-1, r-1) * p^r * q^(x-r) (Distribución binomial negativa)
k2 = 7  # Número total de pozos perforados
r = 3   # Tercer pozo productor
probabilidad_2 = comb(k2 - 1, r - 1) * (p*r) * (q*(k2 - r))
print(f"Probabilidad de que el tercer pozo se produzca en el séptimo pozo perforado: {probabilidad_2:.4f}")

# 3. Media y varianza del número de pozos necesarios para establecer tres pozos productores
# Media: E(X) = r / p
# Varianza: Var(X) = r * (1 - p) / p^2
media = r / p
varianza = r * q / (p**2)
print(f"Media del número de pozos necesarios: {media}")
print(f"Varianza del número de pozos necesarios: {varianza}")


# ## Variables Aleatorias Continuas
# 
# $$ F_X(x) = \mathbb{P}(X\leq x) = \int_{-\infty}^{x}f_X(t)dt$$
# existe $f_X$ que es la función de densidad.
# 
# Podemos encontrar $f_X$ de $X$, como sigue:
# $$ \frac{dF_X(x)}{dx} =f_X(x)$$

# ### Variable aleatoria uniforme sobre el intervalo $(a,b)$.
# 
# Una variable aleatoria $X$ se dice que tiene distribución uniforme continua  en el intervalo $(a,b)$ con $a,b\in\mathbb{R}$, si su función de densidad esta dada por:
# $$f_{X}(x)=\begin{cases}
# 	\frac{1}{b-a} &\quad \text{si } a< x < b\\
# 	\qquad 0 &\quad \text{en otro caso}  
# \end{cases}$$
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim \text{Unif}(a,b)$ $\text{Unif}[a,b]$
# 
# $\color{slateblue}{\text{OBSERVACIÓN.}}$
# * La distribución uniforme continua en $(a,b)$ es simétrica.
# * A los intervalos de la misma longitud contenidos en $(a,b)$ se les asigna la misma probabilidad. Esto se representa gráficamente con la probabilidad de que $X$ se encuentre en el intervalo $(s,t)$:
# 
# 
# La función de distribución de $X\sim Unif((a,b))$ esta dada por:
# $$F_{X}(x)=\begin{cases}
# 	 0 &\quad \text{si } x\le a \\
# 	\frac{x-a}{b-a} &\quad \text{si } a<x<b\\
# 	 1 &\quad \text{si } x\ge b  
# \end{cases}$$
# 
# La esperanza de una variable aleatoria uniforme es la siguiente,
# 
# $$E[X] = \frac{a+b}{2}$$
# 
# Por otro lado, la varianza es la siguiente,
# 
# $$Var(X) = \frac{(b-a)^{2}}{12}.$$

# In[23]:


# Parámetros de la distribución uniforme
a = 0  # límite inferior
b = 1  # límite superior

# Definimos las funciones de densidad y distribución
# Función de densidad (pdf) de la distribución uniforme
def uniform_pdf(x, a, b):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

# Función de distribución (cdf) de la distribución uniforme
def uniform_cdf(x, a, b):
    return np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))
    
# Generar valores de x
x = np.linspace(-0.5, 1.5, 1000)

# Calcular la función densidad 
pdf_values = uniform_pdf(x, a, b)

# Calcular la función de distribución
cdf_values = uniform_cdf(x, a, b)

# Graficar la PDF
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf_values, label='PDF', color='blue')
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.title('Función de Densidad (PDF)')
plt.grid(True)

# Graficar la CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf_values, label='CDF', color='orange')
plt.xlabel('x')
plt.ylabel('Función de distribución acumulada')
plt.title('Función de Distribución Acumulada (CDF)')
plt.grid(True)

plt.tight_layout()
plt.show()


# **Ejemplo.** 	Sea $X\sim Unif((-3,2))$. Vamos a calcular: $P(X\ge 0)$ y
# $P(-5 \le X \le 1/2)$.
# **Solución.** La función de densidad de esta variable aleatoria esta dada por:
# $$f_{X}(x)=\begin{cases}
# 	\quad \frac{1}{5} &\quad \text{si } -3\le x \le 2\\
# 	\quad 0 &\quad \text{e.o.c}  
# \end{cases}$$
# Entonces,
# 
# $$\begin{align}
# \mathbb{P}(X\ge 0) &= \int_{0}^{2}\frac{1}{5}dx=\frac{1}{5}x\Big|_{0}^{2}=\frac{2}{5} \\
# \mathbb{P}(-5 \le X \le 1/2) &= \int_{-3}^{1/2}\frac{1}{5}dx=\frac{1}{5}x\Big|_{-3}^{1/2}=\frac{1}{5}\left(\frac{1}{2}+3\right)=\frac{7}{10}
# \end{align}$$

# $\color{purple}{\text{EJERCICIO 1.}}$ Un alumno se dirige a la biblioteca para solicitar el préstamo de un libro y decide que no puede esperar más de $10$ minutos en ser atendido. Supongamos que el bibliotecario tarda por lo menos $0.5$ minutos en atender a una persona, entonces es razonable proponer una distribución uniforme en el intervalo $[0.5,10]$ para modelar el comportamiento de la variable $X$ que es el tiempo en ser atendido (en  minutos).
# 
# - Da la función de densidad y gráfica.
# - ¿Cuál es la probabilidad de que el tiempo en ser atendido sea mayor a $5$ minutos pero menor a $8$ minutos?
# - ¿Cuál es la esperanza y varianza?
# - Calcula la función de distribución y con ella calcula: $\mathbb{P}(2.51\le X \le 7.99)$.

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Parámetros de la distribución uniforme
a = 0.5  # mínimo
b = 10   # máximo

# Función de densidad y gráfica
x = np.linspace(a - 1, b + 1, 500)
pdf = uniform.pdf(x, loc=a, scale=b-a)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf, 'b-', lw=2, label='Función de densidad')
plt.fill_between(x, pdf, where=(x >= a) & (x <= b), color='blue', alpha=0.2)
plt.title('Función de densidad Uniforme [0.5, 10]')
plt.xlabel('Tiempo de atención (min)')
plt.ylabel('Densidad de probabilidad')
plt.legend()
plt.grid(True)
plt.show()

# Probabilidad entre 5 y 8 minutos
prob_5_8 = uniform.cdf(8, loc=a, scale=b-a) - uniform.cdf(5, loc=a, scale=b-a)
print(f"P(5 < X < 8) = {prob_5_8*100:.2f}%")

# Esperanza y varianza
esperanza = uniform.mean(loc=a, scale=b-a)
varianza = uniform.var(loc=a, scale=b-a)
print(f"E[X] = {esperanza:.2f}")
print(f"Var(X) = {varianza:.4f} ")

# Función de distribución y cálculo de P(2.51 ≤ X ≤ 7.99)
x_cdf = np.linspace(a - 1, b + 1, 500)
cdf = uniform.cdf(x_cdf, loc=a, scale=b-a)

plt.figure(figsize=(10, 5))
plt.plot(x_cdf, cdf, 'r-', lw=2, label='Función de distribución')
plt.title('Función de distribución Uniforme [0.5, 10]')
plt.xlabel('Tiempo de atención (min)')
plt.ylabel('Probabilidad acumulada')
plt.legend()
plt.grid(True)
plt.show()

# Cálculo de la probabilidad
prob_2_51_7_99 = uniform.cdf(7.99, loc=a, scale=b-a) - uniform.cdf(max(2.51, a), loc=a, scale=b-a)
print(f"P(2.51 ≤ X ≤ 7.99) = {prob_2_51_7_99*100:.2f}%")


# In[25]:


from scipy.stats import uniform

# Parámetros de la distribución uniforme
a = 0  # límite inferior
b = 1  # límite superior

# Crear el objeto de distribución uniforme usando scipy
uniform_dist = uniform(loc=a, scale=b-a)

# Generar valores de x
x = np.linspace(-0.5, 1.5, 1000)

# Calcular la PDF usando scipy
pdf_values = uniform_dist.pdf(x)

# Calcular la CDF usando scipy
cdf_values = uniform_dist.cdf(x)

# Graficar la PDF
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf_values, label='PDF', color='blue')
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.title('Función de Densidad (PDF)')
plt.grid(True)

# Graficar la CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf_values, label='CDF', color='orange')
plt.xlabel('x')
plt.ylabel('Función de distribución acumulada')
plt.title('Función de Distribución Acumulada (CDF)')
plt.grid(True)

plt.tight_layout()
plt.show()


# $$\mathbb{P}(X\ge 0) = 1-\mathbb{P}(X<0) = 1-F_X (0)$$
# 
# $$F_X(x):=\mathbb{P}(X\le x)$$
# 
# 
# $$\mathbb{P}(-5 \le X \le 1/2) = F_X (1/2)-F_X(-5)$$

# In[26]:


from scipy.stats import uniform

# Parámetros de la distribución uniforme
a = -3  # límite inferior
b = 2   # límite superior

# Crear el objeto de distribución uniforme usando scipy
uniform_dist = uniform(loc=a, scale=b-a)

# Calcular P(X >= 0) = 1 - P(X < 0)
p_0 = 1 - uniform_dist.cdf(0)

# Calcular P(-5 <= X <= 1/2)
p_interval = uniform_dist.cdf(1/2) - uniform_dist.cdf(-5)

# Imprimir los resultados
print(f"P(X >= 0) = {p_0}")
print(f"P(-5 <= X <= 1/2) = {p_interval}")


# ### Variable aleatoria exponencial con parámetro $\lambda >0$
# 
# La distribución exponencial es una de las distribuciones continuas más utilizadas. A menudo se utiliza para modelar el tiempo transcurrido entre eventos.  
# 
# Algunos ejemplos en los que podría utilizarse la distribución exponencial son:
# * El tiempo transcurrido en un call center hasta recibir la primer llamada del día.
# * El  tiempo entre terremotos de una determinada magnitud.
# * Supongamos una máquina que produce hilo de alambre, la cantidad de metros de alambre hasta encontrar una falla en el alambre se podría modelar como una exponencial.
# 
# Se dice que la variable aleatoria $X$ tiene distribución exponecial de parámetro $\lambda>0$, si su función de densidad está dada por:
# $$f_{X}(x)=\begin{cases}
# \lambda e^{-\lambda x} & x>0\\
# 0 & \text{en otro caso}
# \end{cases}$$
# 
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim \exp(\lambda).$
# 
# $\textbf{Propiedad de pérdida de memoria:}$
# $$\mathbb{P}(X>t+s \mid X>t)=\mathbb{P}(X>s)=e^{-\lambda s}.$$
# 
# $\color{purple}{\text{EJERCICIO.}}$ Demostrar la propiedad de pérdida de memoria para $X\sim \exp(\lambda)$.
# 
# Esta propiedad dice que la probabilidad de que el tiempo adicional que tienes que esperar sea mayor que $t$, dado que ya has esperado $s$, es igual a la probabilidad original de esperar más de $t$. Es como si el proceso "olvidara" que ya pasó tiempo $s$.
# 
# $\color{purple}{\text{DEMOSTRACIÓN.}}$
# Queremos demostrar que si $X \sim Exp(\lambda)$ con $\lambda > 0$, entonces cumple la propiedad de pérdida de memoria antes mencionada.
# 
# Para $X \sim \text{Exp}(\lambda)$, la función de distribución acumulada (CDF) es
# 
# $$F_X(x) = 1 - e^{-\lambda x}, \quad x \geq 0$$
# 
# Por lo tanto, la esperanza de x es
# 
# $$E[x] = e^{-\lambda x}$$
# 
# Usamos la definición de probabilidad condicional
# 
# $$P(X > s + t \mid X > s) = \frac{P(X > s + t \cap X > s)}{P(X > s)}$$
# 
# Como $X > s + t$ implica automáticamente que $X > s$, tenemos
# 
# $$P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)}$$
# 
# Sustituyendo en la esperanza de x
# 
# $$P(X > s + t \mid X > s) = \frac{e^{-\lambda (s + t)}}{e^{-\lambda s}} = e^{-\lambda t}$$
# 
# De manera que
# 
# $$ P(X > s + t \mid X > s) = e^{-\lambda t} = P(X > t)$$
# 
# Por lo tanto, se demuestra que la distribución exponencial cumple la propiedad de **pérdida de memoria**:
# 
# $$P(X > s + t \mid X > s) = P(X > t)$$
# 

# $\color{indigo}{\text{EJEMPLO.}}$ Hipótesis natural para modelar las duraciones de vida de átomos radioactivas (Rutherford y Soddy). Cada átomo radioactivo posee una duración de vida que sigue una ley exponencial. En este campo, el parámetro $\lambda$ se llama la constante de desintegración.
# 
#  Si $t\mapsto \rho(t)=\mathbb{P}(X>t)$ verifica
# $$\rho(t+s)=\rho(t)\rho(s),$$
# de manera que (derivando en $s$, con $s=0$),
# $$\rho^{\prime}(t)=-\rho(t)\lambda \qquad \ \lambda=-\rho^{\prime}(0)\geq 0.$$
# Así,
# $$\rho(t)=e^{-\lambda t} \qquad \text{y} \qquad f(t)=\lambda e^{-\lambda t} \ \  \rho(0)=1.$$
# 
# La esperanza y varianza de de una distribución exponencial de la forma $$f_{X}(x)=\begin{cases}
# \frac{1}{\lambda} e^{-\frac{x}{\lambda}} & x>0\\
# 0 & \text{eoc}
# \end{cases}$$
# 
# $$E[X] = \lambda$$
# y
# $$\text{Var}(X) = \lambda^{2}$$
# 
# $\color{indigo}{\text{EJEMPLO.}}$
# 
# Consideremos la variable aleatoria $X$ como el tiempo (en minutos) entre la llegada de dos personas a la fila  de una sucuarsal bancaria.
# 
# Adicionalmente, el banco ha determinado que solo el $10\%$ de las veces, el tiempo que transcurre entre la llegada de una persona y otra es mayor a dos minutos.
# 
# Esto permite calcular el valor de $\lambda$, ya que
# $$\mathbb{P}(X>2)=0.1$$
# entonces
# $$1-\mathbb{P}(X\le 2) = 1-F_{X}(2) = 0.1$$
# 
# Por lo que $F_{X}(2)= 1-e^{\frac{-2}{\lambda}} =0.9$, entonces
# $e^{\frac{-2}{\lambda}}=0.1$. Por lo que
# $$\frac{-2}{\lambda}=\ln(0.1) \Rightarrow \lambda=0.87$$
# Por lo tanto $X\sim\exp(0.87)$.
# 
# Ahora queremos calcular la probabilidad de que entre la llegada de una persona y otra transcurra por lo menos un minuto, lo cual puede calcularse de dos formas:
# 
# 
# \begin{align*}
# \mathbb{P}(X>1) &= \int_{1}^{\infty}f_{X}(x)dx = \int_{1}^{\infty}\frac{1}{0.87}e^{\frac{-x}{0.87}}dx = 0.32
# \end{align*}
# \item \begin{align*}
# \mathbb{P}(X>1) &= 1-\mathbb{P}(X\le 1) = 1-\left[1-e^{\frac{-1}{0.87}}\right] = -e^{\frac{-1}{0.87}}
# \end{align*}

# Una propiedad interesante de la distribución exponencial es que puede verse como un análogo continuo de la distribución geométrica. Para ver esto, recuerde el experimento aleatorio detrás de la distribución geométrica: lanza una moneda (repite un experimento de Bernoulli) hasta que observa las primeras caras (éxito).
# 
# 
# $\color{slateblue}{\text{TEOREMA.}}$ Sea $\varepsilon>0$ y $Y_{\epsilon}\sim Geo(p_{\varepsilon})$. Supongamos que nos encontramos en el regimen:
# $$\lim_{\varepsilon\to 0}p_{\varepsilon}=0 \qquad \text{y} \qquad \varepsilon^{-1}p_{\varepsilon}\sim \lambda>0$$
#    Sea $X_{\varepsilon}:=\varepsilon Y_{\varepsilon}$. Entonces,
#    $$\lim_{\varepsilon\to 0}F_{X_{\varepsilon}}(x)=F_{X}(x),$$
#    en donde $X\sim \exp(\lambda)$. Este también es un resultado de convergencia en ley.
# 
# $\color{slateblue}{\text{TEOREMA.}}$ Sea $X\sim \exp(1)$. Si $Y=\min\{k\in \mathbb{Z}: k\geq \lambda X \}$, con $\lambda>0$, entonces
# $$Y\sim Geo\left(p=1-e^{-1/\lambda} \right).$$

# ### Variable aleatoria normal con parámetros media $\mu$ y varianza $\sigma^{2}$
# 
# La distribución normal es una de las más importantes y de mayor uso tanto en la teoría de la probabilidad, como en la teoría estadística.
# 
# También llamada distribución gaussiana, en honor a Gauss, a quien se considera el padre de ésta distribución.
# 
# La importancia de la distribución normal, radica en el famoso Teorema central del límite. Fue descubierta por De Moivre en 1733 como un límite de la distribución binomial.
# 
# 
# La importancia de esta distribución radica en que permite modelar numerosos fenómenos naturales, sociales y psicológicos, por ejemplo:
# * Estatura
# * Efectos de un fármaco
# * Consumo de cierto producto por un grupo de individuos
# * Coeficiente intelectual
# * Nivel de ruido en telecomunicaciones
# * Errores cometidos al medir ciertas magnitudes
# 
# Además, esta distribución juega un papel de suma importancia en la inferencia estadística.
# 
# Se dice que la variable aleatoria $X$ tiene distribución normal de parámetros $\mu$ y $\sigma^{2}$, donde $\mu,\sigma\in\mathbb{R}$ y $\sigma>0$, si su función de densidad está dada por:
# $$f_{X}(x)=\begin{cases}
# 	\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} &\quad \text{si }  x \in\mathbb{R} \\
# 	\qquad 0 &\quad \text{e.o.c}  
# \end{cases}$$
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim N(\mu,\sigma^{2})$
# 
# 
# Tal curva (**la campana de Gauss-Bell**) es una función que depende de los parámetros $\mu$ y $\sigma^{2}$.
# 
# 
# $\color{slateblue}{\text{OBSERVACIÓN.}}$ El parámetro $\mu$ se llama *media* y el  parámetro $\sigma^{2}$ se llama *varianza*.
# 
# La función de densidad definida anteriormente es efecto función de densidad, ya que
# $$\int_{\mathbb{R}}f_{X}(x)dx = \int_{\mathbb{R}}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-m)^{2}}{2\sigma^{2}}}  = 1 $$
# 
# Variable aleatoria normal con parámetros  0  y  1 .
# La ley de $Z\sim N(0,1)$ está dada por la función de densidad
# $$
# f_{Z}(z) = \frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}},
# $$
# para $z\in \mathbb{R}$.
# 
# Sabemos que:
# * La varianza es usada como una medida para comparar la dispersión en dos o más conjuntos de observaciones.
# * Una desviación estándar pequeña indica que los valores de la variable aleatoria se encuentran cercanos a la media.
# * Una desviación estándar grande indica que los valores de la variable aleatoria se dispersan mucho con respecto a la media.
# 
# La función de distribución de una variable aleatoria $X\sim N(\mu,\sigma^{2})$ está dada por:
# $$F_{X}(x) = \int_{-\infty}^{x}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y-\mu)^{2}}{2\sigma^{2}}}dy$$
# 
# Esta nos proporciona la probabilidad de que $X$ tome calores menores o iguales a un valor específico $x$, y corresponde al área bajo la curva en el intervalo $(-\infty,x]$:
# 
# No es sencillo calcular $F_{X}(x)$, pero cualquier v.a. gaussiana puede transformarse a una v.a. estandarizada. Existen tablas para esta v.a., lo cual hace los cálculos más fáciles.
# 
# $\color{slateblue}{\text{PROPOSICIÓN.}}$  Sea $X\sim N(\mu,\sigma^{2})$, entonces
# $$Z=\frac{X-\mu}{\sigma}$$
# tiene una distribución gaussiana con media $0$ y varianza $1$, es decir, $Z\sim N(0,1)$.
# 
# 
# $\color{indigo}{\text{DEMOSTRACIÓN}}$ 	Primero vamos a calcular la función de distribución de $Z$:
# $$F_{Z}(z):=\mathbb{P}(Z\le z) = \mathbb{P}\left(\frac{X-\mu}{\sigma}\le z\right) = \mathbb{P}(X\le \sigma z + \mu) := F_{X}(\sigma z + \mu)$$
# Entonces, la función de densidad de $Z$ esta dada por:
# $$f_{Z}(z) = \frac{dF_{Z}(z)}{dz} = \frac{dF_{X}(\sigma z + \mu)}{dz} = \sigma f_{X}(\mu+\sigma z) = \frac{e^{-z^{2}/2}}{\sqrt{2\pi}}$$
# 
# Decimos que $X$ tiene distribución gaussiana estándar (normal estándar), $X\sim N(0,1)$, si su función de densidad esta dada por:
# $$f_{X}(x)=\begin{cases}
# 	\frac{e^{-x^{2}/2}}{\sqrt{2\pi}} &\quad \text{si }  x \in\mathbb{R} \\
# 	\quad 0 &\quad \text{e.o.c}  
# \end{cases}$$
# 
# Podemos calcular cualquier probabilidad de la forma
# $$\mathbb{P}(a<X<b)$$
# de la siguiente manera: para $z\ge 0$, definimos
# $$\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{0}^{z}e^{-x^{2}/2}dx.$$
# 

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
fig, ax = plt.subplots()
x= np.arange(-4,4,0.001) #generar valores de x
ax.set_title('N(0,$1^2$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.plot(x, norm.pdf(x))
ax.set_ylim(0,0.45)
plt.show()


# Propiedades de la función de densidad de probabilidades de una distribución normal estándar:
# 
# 1. Es positiva: $f(x)\geq 0$ para todo $x$ real.
# 2. Es continua y derivable en todas partes.
# 3. Es simétrica alrededor de $\mu$.
# 4. Conforme $x$ toma valores muy grandes de manera positiva y negativa, la función decrece hacia cero muy rápidamente.
# 5. Tiene un máximo global.
# 6. El área total bajo la curva es igual a $1$.

# Veamos el comportamiento de la función conforme se cambia la varianza.

# In[28]:


from scipy.stats import norm
fig, ax = plt.subplots()
x = np.linspace(-10,10,100)
stdvs = [0.5, 0.7, 1.0, 2.0, 3.0, 4.0]
for s in stdvs:
    ax.plot(x, norm.pdf(x,scale=s), label='stdv=%.1f' % s)

ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.set_title('Distribución normal')
ax.legend(loc='best', frameon=True)
ax.set_ylim(0,1)
ax.grid(True)


# Veamos el comportamiento de la función conforme se cambia la media.

# In[29]:


from scipy.stats import norm
fig, ax = plt.subplots()
x = np.linspace(-10,10,100)
means = [-1.0,-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
for mean in means:
    ax.plot(x, norm.pdf(x,loc=mean), label='mean=%.1f' % mean)

ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.set_title('Distribución normal')
ax.legend(loc='best', frameon=True)
ax.set_ylim(0,0.45)
ax.grid(True)


# ### Función de distribución acumulativa de una normal $N(\mu,\sigma^2)$
# 
# Gracias a las propiedades anteriores, es posible calcular áreas delimitadas de la función $f$. Si $a$ y $b$ son reales cualesquiera, denotaremos por
# $$P(a\leq X\leq b),$$
# la probabilidad de que $X$ esté en el intervalo $[a,b]$, al área bajo la curva de $f(x)$ sobre el intervalo $[a,b]$.
# 
# También, $P(X\leq x)$ denotara al área bajo la curva de la función $f(x)$ sobre el intervalo $(-\infty,x)$ y $P(X>x)$ denotara al área bajo la curva de la función $f(x)$ sobre el intervalo $(x, \infty)$.

# A la probabilidad $\text{cdf}(x):=P(X\leq x)$ se llama la distribución acumulativa (hasta el valor $x$) de $f(x)$.
# 
# Con la notación anterior,
# $$P(a\leq X\leq b)=\text{cdf}(b)-\text{cdf}(a)$$
# y
# $$\text{sf}(a):=P(X>a)=1-\text{cdf}(a).$$

# In[30]:


from scipy.stats import norm
fig, ax = plt.subplots()
# for distribution curve
x= np.arange(-4,4,0.001)
ax.plot(x, norm.pdf(x))
ax.set_title("Distribución normal acumulativa")
ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.grid(True)
# for fill_between
px=np.arange(-4,1,0.01)
ax.set_ylim(0,0.5)
ax.fill_between(px,norm.pdf(px),alpha=0.5, color='g')
# for text
ax.text(-1,0.1,"cdf(x)", fontsize=20)
plt.show()


# ### Cálculo de probabilidades de una distribución normal
# 
# Calculemos $\text{cdf}(2)=\mathbb{P}(X<2)$ cuando $X\sim N(3,2^2)$.
# 
# $$Z = \frac{X-\mu}{\sqrt{\sigma^2}} \sim N(0,1)$$
# 

# In[31]:


norm.cdf(x=2, loc=3, scale=2) #v.a. con media=3 y st=2


# In[32]:


from scipy.stats import norm
fig, ax = plt.subplots()
# for distribution curve
x= np.arange(-4,10,0.001)
ax.plot(x, norm.pdf(x,loc=3,scale=2))
ax.set_title("N(3,$2^2$)")
ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.grid(True)
# for fill_between
px=np.arange(-4,2,0.01)
ax.set_ylim(0,0.25)
ax.fill_between(px,norm.pdf(px,loc=3,scale=2),alpha=0.5, color='g')
# for text
ax.text(-0.5,0.02,round(norm.cdf(x=2, loc=3, scale=2),2), fontsize=20)
plt.show()


# Calculemos $\mathbb{P}(0.5<𝑋<2)$ cuando $X\sim N(1,2)$.

# In[33]:


s=np.sqrt(2)
norm(1, s).cdf(2) - norm(1,s).cdf(0.5)


# In[34]:


fig, ax = plt.subplots()
x= np.arange(-6,8,0.001)
ax.plot(x, norm.pdf(x,loc=1,scale=2))
ax.set_title("N(1,$2^2$)")
ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.grid(True)
px=np.arange(0.5,2,0.01)
ax.set_ylim(0,0.25)
ax.fill_between(px,norm.pdf(px,loc=1,scale=2),alpha=0.5, color='g')
pro=norm(1, 2).cdf(2) - norm(1,2).cdf(0.5)
ax.text(0.2,0.02,round(pro,2), fontsize=20)
plt.show()


# $\color{indigo}{\text{EJEMPLO.}}$ Si $Z\sim N(0,1)$, encuentra $\mathbb{P}(-1.93 < Z < 1.93)$. La probabilidad buscada es:

# In[35]:


norm(0,1).cdf(1.93)-norm(0,1).cdf(-1.93)


# In[36]:


fig, ax = plt.subplots()
# for distribution curve
x= np.arange(-3,3,0.001)
ax.plot(x, norm.pdf(x,loc=0,scale=1))
ax.set_title("N(0,$1^2$)")
ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
ax.grid(True)
px=np.arange(-1.93,1.93,0.01)
ax.set_ylim(0,0.45)
ax.fill_between(px,norm.pdf(px,loc=0,scale=1),alpha=0.5, color='g')
pro=norm(0, 1).cdf(1.93) - norm(0,1).cdf(-1.93)
ax.text(0.2,0.02,round(pro,2), fontsize=20)
plt.show()


# ### Cálculo de probabilidades:
# $\mathbb{P}(0<Z<b)$: Queremos calcular $\mathbb{P}(0<Z<0.43)$, lo cuál puede realizarse de la siguiente manera:
# * Tablas de área a la derecha: $\mathbb{P}(0<Z<0.43) = 0.1664$
# 
# $\mathbb{P}(-b<Z<b)$: Queremos calcular $\mathbb{P}(-0.16<Z<0.16)$, lo cuál puede realizarse de la siguiente manera:
# * Tablas de área a la derecha:  $\mathbb{P}(-0.16<Z<0.16) = \mathbb{P}(-0.16<Z<0)+\mathbb{P} (0<Z<0.16) = \mathbb{P}(0<Z<0.16)+\mathbb{P}(0<Z<0.16) = 2\mathbb{P}(0<Z<0.16) = 2(0.0636) = 0.1272$
# 
# $\mathbb{P}(Z<-b)$: Queremos calcular $\mathbb{P}(Z<-1.94)$, lo cuál puede realizarse de la siguiente manera:
# * Tablas de área a la derecha: $\mathbb{P}(Z<-1.94) = \mathbb{P}(Z<0)+\mathbb{P}(-1.94<Z<0) = \mathbb{P}(Z<0) + \mathbb{P}(0<Z<1.94) = 0.5 - 0.4738 = 0.0262$
# 
# $\mathbb{P}(Z>-b)$: Queremos calcular $(Z>-0.07)$, lo cuál puede realizarse de la siguiente manera:
# * Tablas de área a la derecha: $\mathbb{P}(Z>-0.07) = \mathbb{P}(-0.07<Z<0) + \mathbb{P}(Z>0) = \mathbb{P}(0<Z<0.07) + \mathbb{P}(Z>0) = 0.0279 + 0.5 = 0.5279$

# ### Variable aleatoria Gamma con parámetros $\alpha$ y $\lambda$
# 
#  La distribución gamma se obtiene al considerar el tiempo que transcurre entre cierto número de ocurrencias de eventos que ocurren aleatoriamente en el tiempo
# 
# La función gamma $\Gamma:(0,\infty)\to \mathbb{R}$ está definida como
# 	$$\Gamma(\alpha)=\int_{0}^{\infty}t^{\alpha-1}e^{-t}dt.$$
# 
# 
# 
# *Propiedades de la función gamma*.
# - $\Gamma(\alpha)<\infty$ para cualquier $\alpha>0$.
# - $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$.
# - Si $n\geq 1$ $\Gamma(n)=(n-1)!$
# - $\Gamma\left(\frac{1}{2} \right)=\sqrt{\pi}$.
# 
# 
# 
#  Si $\alpha$ y $\lambda$ son reales positivos, decimos que una variable aleatoria $X$ tiene distribución gamma con parámetros $\alpha$ y $\lambda$ si tiene por función de densidad:
# 	$$\displaystyle f_{X}(x)=\begin{cases}
# 	\displaystyle \frac{\lambda^{\alpha}x^{\alpha-1}e^{-\lambda x} }{\Gamma(\alpha)} & x>0\\
# 	0 & \text{en otro caso}
# 	\end{cases}$$
# En esta caso, escribimos la información anterior como $X\sim \Gamma(\alpha,\lambda) $
# 
# 
# $\color{indigo}{\text{EJEMPLO 1.}}$ Una computadora cuántica cuenta con un tipo de aparato de medición, el cual tiene un tiempo de vida que se distribuye exponencialmente, de tal manera que su tiempo promedio de vida es de  $1000$ horas. Si se utilizan $10$ de estos aparatos en forma consecutiva, uno de ellos después de que el anterior ya no funciona. ¿Cuál es la probabilidad de que alguno de los aparatos estará funcionando después de $10,000$ horas?
# 
# 
# $\color{indigo}{\text{SOLUCIÓN.}}$
# Sea $X$ el tiempo total de vida de los $5$ aparatos, usados, como se indica, uno después del otro. Entonces $X\sim \Gamma(10, 0.001)$. Así,
# $$\mathbb{P}(X>10000)=\int_{10000}^{\infty}\frac{(0.001)^{10}}{9!}x^{9}e^{-0.001x}dx=0.4579.$$
# 
# $\color{indigo}{\text{EJEMPLO 2.}}$  Consideremos un call center donde los tiempos entre llamadas son independientes y se distribuyen exponencialmente con una media de 3 minutos. Supongamos que queremos encontrar la probabilidad de que transcurran más de 30 minutos antes de recibir 10 llamadas.
# 
# 
# $\color{indigo}{\text{SOLUCIÓN.}}$
# Sea $X$ el tiempo total para recibir $10$ llamadas. Dado que el tiempo medio entre llamadas es de $3$ minutos. Entonces $X \sim \Gamma(10, \frac{1}{3})$. Así
# $$P(X > 30) = \int_{30}^{\infty} \frac{\left(\frac{1}{3}\right)^{10} x^{9} e^{-\frac{x}{3}}}{9!} \, dx = 0.45793.$$
# 
# $\color{slateblue}{\text{NOTA.}}$ Si $X\sim N(0,1)$, entonces $X^{2}\sim \Gamma\left(\frac{1}{2}, \frac{1}{2} \right)$.
# En efecto, Para $z>0$, se tiene:
# 	$$F_{X^{2}}(z)=\mathbb{P}(X^2\leq z)=\mathbb{P}\left(-\sqrt{z}\leq X\leq \sqrt{z}\right)=F_{X}\left(\sqrt{z}\right)-F_{X}\left(-\sqrt{z}\right).$$
# Por lo tanto,
# $$f_{X^2}(z)=\frac{d F_{X^{2}}(z)}{dz}=\frac{1}{2\sqrt{z}}f_{X}\left(\sqrt{z}\right)+\frac{1}{2\sqrt{z}}f_{X}\left(-\sqrt{z}\right)=\frac{1}{\sqrt{z}}f_{X}(z).$$
# Ahora,
# $$f_{X}\left(\sqrt{z}\right)=\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z}.$$
# 
# 
# ### Esperanza y Varianza
# 
# La esperanza, $\text{E}[X]$ para una variable aleatoria $X\sim Γ(\alpha, \lambda)$ es:
# 
# $$E(X) = \frac{\alpha}{\lambda}$$
# 
# y su varianza es:
# 
# $$\text{Var}(X) = \frac{\alpha}{\lambda^2}$$
# 
# La distribución gamma es esencial en varios campos por su capacidad para modelar tiempos de espera y eventos con tasas constantes. Sus aplicaciones incluyen:
# 
#   * $\color{indigo}{\text{Teoría de colas y procesos estocásticos:}}$ Modela el tiempo de espera hasta el k-ésimo evento.
#   * $\color{indigo}{\text{Confiabilidad y análisis de supervivencia:}}$ Utilizada para tiempos de fallo y eventos críticos en medicina.
#   * $\color{indigo}{\text{Hidrología y meteorología:}}$ Aplica en la modelación de precipitaciones acumuladas y tamaños de gotas.
#   * $\color{indigo}{\text{Procesamiento de imágenes y señales:}}$ En el ajuste de modelos a datos de intensidades.
#   * $\color{indigo}{\text{Finanzas:}}$ Para rendimientos de activos que no siguen distribuciones normales.
#   * $\color{indigo}{\text{Biología y ecología}}$ Para tasas de crecimiento y tiempos entre eventos biológicos.
#   * $\color{indigo}{\text{Física:}}$ Describe tiempos de decaimiento y distribuciones de energía.
# 

# - Si $\alpha = 1$, $\lambda > 0 \Rightarrow$ v.a. exponencial.
# 
# - Si $\lambda = \frac{1}{2}$, $\alpha = \frac{k}{2}$, $k \in \mathbb{Z}^{+} \Rightarrow$ ji-cuadrada.
# 
# - Si $\lambda > 1$ y $\alpha > 1 \Rightarrow$ Erlang $\Rightarrow$ aplicaciones.
# 

# $\color{indigo}{\text{EJEMPLO.}}$ Sea $X \sim \mathcal{N}(0,1)$. Demostrar que $ X^2 $ tiene una distribución gamma.
# 
# $\color{indigo}{\text{SOLUCIÓN.}}$ Para $z \geq 0$, se tiene que:
# 
# $$
# \begin{align*}
# F_{X^2}(z) &= P(X^2 \leq z) = P(-\sqrt{z} \leq X \leq \sqrt{z}) \\
# &= F_X(\sqrt{z}) - F_X(-\sqrt{z}) \\
# \end{align*}
# $$
# Derivando, obtenemos la función de densidad:
# 
# $$ f_{X^2}(z) = \frac{d}{dz} F_{X^2}(z) = \frac{1}{2\sqrt{z}} f_X(\sqrt{z}) + \frac{1}{2\sqrt{z}} f_X(-\sqrt{z}) $$
# 
# Como la densidad de $X$ es simétrica, se tiene que 
# 
# $$ f_{X^2}(z) = \frac{1}{\sqrt{2\pi}} e^{-z/2} \frac{1}{2\sqrt{z}} $$
# 
# $$ \frac{(1/2)^{1/2} z^{-1/2} e^{-z/2}}{\Gamma(1/2)} $$
# 
# $$ \sim Gamma\left(\frac{1}{2}, \frac{1}{2}\right) $$

# ### Variable aleatoria Beta con parámetros $\alpha$ y $\beta$
# 
# 
# La distribución beta es una familia de distribuciones de probabilidad continua definida en el intervalo [0, 1]. Es particularmente útil para modelar variables que representan proporciones y porcentajes.
# 
# La función beta, $B(\alpha, \beta)$, se define como:
# 
# $$
# B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1} dt = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}
# $$
# 
# La función de densidad de probabilidad de la distribución beta se expresa como:
# 
# $$
# f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)},
# $$
# con $x\in (0,1)$.
# 
# $\color{slateblue}{\text{NOTACIÓN.}}$ $X\sim \text{Beta}(\alpha, \beta)$.
# 
# Si $X\sim \text{Beta}(\alpha, \beta)$, entonces
#   $$E[X] = \frac{\alpha}{\alpha + \beta}$$
# y
#   $$\text{Var}(X) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}.$$
# 
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Supongamos que un análisis sugiere que una nueva inversión tiene una alta probabilidad de éxito. Usando una distribución beta con $\alpha = 5$ y $\beta = 1$, entonces:
# 
# $$f_{X}(x) = 5x^4, \quad E[X] = \frac{5}{6}, \quad \text{Var}(X) = \frac{5}{252}$$
# 
# 
# Aplicaciones en la Vida Real
# 
# La distribución beta se utiliza en una variedad de campos, incluyendo:
# - $\color{indigo}{\text{Finanzas:}}$ para modelar la variabilidad en tasas de retorno de inversiones.
# - $\color{indigo}{\text{Mercadotecnia:}}$ para analizar proporciones de respuesta de consumidores.
# - $\color{indigo}{\text{Ciencias de la salud:}}$ en la evaluación de la efectividad de tratamientos médicos.
# - $\color{indigo}{\text{Ecología:}}$ para estimar proporciones en estudios de biodiversidad.

# ## Simulación de una Variable Aleatoria
# 
# ### El problema del encuentro 
# 
# Romeo y Julieta pretenden encontrarse en un lugar específico entree $[0,60]$.
# 
# - Si Romeo llega primero, entonces piensa esperar 10 minutos
# - Si Julieta llega primero, entonces piensa esperar 10 minutos
# 
# Se elige un número al azar entre $0$ y $60$, y supongamos que el par de números se elige sobre el cuadrado $60^2$, de modo que la probabilidad de un evento, sea el área favorable entre el área total.
# 
# Sea el evento 
# - $A$: Romeo y Julieta se encuentran a una distancia de $10$ minutos. 

# In[37]:


import matplotlib.pyplot as plt
import numpy as np

# Vamos a generar 100,000 simulaciones
muestra = 1000000

# Generamos los tiempos aleatorios entre [0,60]
# x : tiempo de llegada de Romero [0,60]
# y : tiempo de llegada de Romero [0,60]

x,y = np.random.uniform(0,60,muestra), np.random.uniform(0,60,muestra)

# Vamos a añadir nuestra condición de encuentro 
encuentro = np.where(abs(x-y)<=10, 'r', 'b') # r si se encuentran y b si no se encuentran

# Calculo de la probabilidad del encuentro
k=0
for elemento in encuentro:
    if elemento == 'r':
        k += 1

print("La probabilidad del evento A es: {}.".format(k/muestra))

plt.scatter(x,y,c=encuentro,s=1)
plt.show()


# ### El problema de Monty Hall
# 
# Es un famoso dilema de probabilidad basado en un concurso de televisión, y es un gran ejemplo de cómo las intuiciones humanas pueden ser engañosas en contextos probabilísticos.
# 
# #### Planteamiento del problema:
# 
# Tienes tres puertas enfrente de ti. Detrás de una de ellas hay un coche (el premio), y detrás de las otras dos hay cabras (premios no deseados).
# 
# Tu elección inicial: Eliges una puerta, pero no la abres aún.
# 
# Monty Hall (el presentador, que sabe qué hay detrás de cada puerta) abre una de las otras dos puertas, revelando una cabra.
# 
# Segunda oportunidad: Ahora Monty te da la opción de cambiar de puerta (de tu elección original a la otra puerta que queda cerrada) o quedarte con tu elección inicial.
# 
# **¿Qué deberías hacer? ¿Cambiar de puerta o quedarte con tu elección original?**
# 
# * Respuesta correcta: Deberías cambiar de puerta.
# * Explicación: Inicialmente, al elegir una puerta, tienes una probabilidad de $1/3$ de haber elegido el coche y una probabilidad de $2/3$ de haber elegido una cabra.
# 
# Cuando Monty abre una puerta revelando una cabra, no cambia estas probabilidades. Al principio tenías un 2/3 de probabilidades de haber elegido mal, y Monty, al mostrar una cabra, te está ayudando a confirmar que el coche probablemente está detrás de la otra puerta.
# 
# Si elegiste mal inicialmente (lo cual sucede en el 66.67% de los casos), Monty te da la oportunidad de corregir tu elección cambiando a la puerta correcta.
# 
# Si elegiste correctamente desde el principio (probabilidad del 33.33%), entonces cambiar haría que perdieras.
# 
# Por lo tanto, cambiar de puerta te da una probabilidad de ganar de $2/3$, mientras que quedarte con tu elección original te deja con solo una probabilidad de ganar de $1/3$.
# 
# $\color{indigo}{\text{EJEMPLO.}}$
# 
# * Puertas: [1, 2, 3]
# * El coche está detrás de la puerta 2.
# * Si eliges la puerta 1 (probabilidad $1/3$), Monty abre la puerta 3, que tiene una cabra.
# * Si cambias a la puerta 2, ganas el coche (probabilidad $2/3$).

# #### Implementación en Python
# 
# El siguiente código simula el famoso problema de Monty Hall en tres escenarios distintos:
# 
# * Mantener la elección inicial (no cambiar de puerta).
# * Cambiar siempre de puerta.
# * Elegir de manera aleatoria entre mantener o cambiar.

# 1.- Generar el juego
# La función generate_game(n) crea un conjunto de juegos. Cada juego tiene 3 puertas, una de ellas con un premio (el coche) y las otras dos con una cabra.

# In[38]:


from random import randint
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Esta función generará rondas aleatorias para nuestro juego. Cada ronda consta de 3 puertas. Solo una de las puertas es correcta, las otras dos son incorrectas.
# Esta función genera un conjunto de n juegos, cada uno con tres puertas, y solo una contiene el coche.

def generate_game(n: int):
    game = []

    for _ in range(n):
        doors = [False] * 3 # Inicializa 3 puertas como "falsas", indicando que no tienen el premio.
        winner = randint(0, 2) # Escoge aleatoriamente cuál puerta tiene el coche (premio).
        doors[winner] = True  # La puerta seleccionada se marca como True (es la puerta ganadora).
        game.append(doors) # Añade este set de puertas al juego.

    return game


# Esta es una función auxiliar que toma una lista de 3 puertas, examina la segunda y la tercera puerta y luego abre la que tiene una cabra (es decir, la puerta equivocada).
#Esto simula un host con conocimiento de lo que hay detrás de las puertas.
def reveal_goat(doors):
    # Get from doors 2 and 3 the one which contains goat.
    for i in range(1, 3): # Revisa solo las puertas 2 y 3, porque la puerta 1 es la que elige inicialmente el jugador.
        if doors[i] == False: # Si la puerta no contiene el coche.
            return i  # Devuelve el índice de la puerta con cabra.


# ***Simular una elección aleatoria**
#
# Simular una situación en la que el jugador elige aleatoriamente si desea mantener su elección inicial o cambiarla.

def simulate_random_choice(game: list): #El jugador decide al azar si cambia o no después de que Monty revele una cabra.
    wins = 0
    attempts = 0

    history = []

    for doors in game:
        attempts += 1

        #Monty revela una cabra.
        goat = reveal_goat(doors)

        # El jugador decide aleatoriamente si cambiar o no.
        new_choice = randint(0, 1)
        final_choice = 0 if new_choice == 0 else 2 if goat == 1 else 1 # Calcula la nueva elección

        if (doors[final_choice] == True): #Si la elección es correcta, suma un win
            wins += 1

        history.append(wins / attempts) # Añade el ratio de victorias hasta el momento.

    return wins, history


# **Simular elección inicial**
#
# Simular una situación en la que el jugador *solo* conserva su elección inicial y nunca cambia

def simulate_keep_choice(game: list): #El jugador nunca cambia de puerta.
    wins = 0
    attempts = 0
    history = []

    for doors in game:
        attempts += 1

        # User does not switch game.
        if (doors[0] == True):  # Si la primera puerta (la inicial) tiene el coche, suma un win.
            wins += 1

        history.append(wins / attempts)  # Registra el ratio de victorias.

    return wins, history


# **Simular elección de cambio**
#
# Simular una situación en la que el jugador cambia su elección cada vez.

# El jugador cambia de puerta siempre después de que Monty revele una cabra.

def simulate_switch_choice(game: list):
    wins = 0
    attempts = 0
    history = []

    for doors in game:
        attempts += 1

        # Monty revela una cabra.
        goat = reveal_goat(doors)

        # Player switches his doors (here he chooses the non-opened doors).
        new_choice = 1 if goat == 2 else 2 # El jugador cambia a la otra puerta no abierta.

        if (doors[new_choice] == True): # Si la nueva elección es correcta, suma un win.
            wins += 1

        history.append(wins / attempts) # Registra el ratio de victorias.


    return wins, history


# Ahora comienza el cálculo, que genera $n$ juegos aleatorios para la simulación.

# El código genera un conjunto de 1000 juegos y ejecuta las tres simulaciones:
game = generate_game(1000)


# Ejecute las tres simulaciones definidas anteriormente para el juego generado.

wins_random, history_random = simulate_random_choice(game)
wins_keep, history_keep = simulate_keep_choice(game)
wins_switch, history_switch = simulate_switch_choice(game)


# Y por último, crea un gráfico para que podamos ver el resultado.
plt.figure(figsize=(12,8))
plt.plot(history_random, 'r', label="Cambio aleatorio")
plt.plot(history_keep, 'g', label="Mantener la inicial")
plt.plot(history_switch, 'b', label="Solo cambiar")
plt.legend(loc='upper right')
plt.ylim(0, 1.0)
plt.xlim(0, 1000)
plt.ylabel("Probabilidad", fontsize=16)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.xlabel("Iteraciones", fontsize=16)
plt.grid(True)
plt.show()


# Se visualiza cómo cambian las probabilidades de éxito para las dos estrategias principales: mantener la elección inicial y cambiar de puerta.

# ### Generación y uso de variables aleatorias

# Daremos recopilación de métodos para generar variables aleatorias utilizados en simulación computacional. Se abordan distribuciones uniformes, exponenciales, normales, binomiales y Poisson, entre otras, así como técnicas como la transformada inversa y el método del rechazo.
# 
# Ejemplos comunes en simulación:
# - Tiempo entre llegada de cada persona  
# - Número de personas por minuto  
# - Número de artículos por persona  
# - Cantidad de dinero ganado cada hora  
# - Tiempo de atención por cliente  
# - Número de veces que la cajera solicita ayuda durante la jornada  
# - Cantidad de gente que está formada  
# - Número de personas que pagan con tarjeta

# ## Métodos de generación de números aleatorios rectangulares
# 
# La generación de variables aleatorias con esta distribución es importante porque las variables que tengan una distribución diferente, tendrán que usar a ésta como base.
# 
# - Cada posible resultado entre $a$ y $b$ tiene la misma probabilidad $1/n$.
# 
# Las variables generadas deben cumplir con:
# - Los valores generados deben ser independientes y estar idénticamente distribuidos
# - La secuencia generada debe ser lo más larga posible y ser reproducibles
# - Debe permitir generar múltiples secuencias
# - Que usen poca memoria
# 
# Históricamente se han usado cuatro tipos de métodos para generar sucesiones de números rectangulares:
# - Métodos manuales
# - Tablas de biblioteca
# - Computación analógica
# - Computación digital

# ### Generación pseudoaleatoria
# 
# La generación de los números aleatorios rectangulares debe realizarse a través de relaciones matemáticas de recurrencia. Por esta razón se consideran **pseudoaleatorios**, ya que el proceso para generarlos es determinístico.
# 
# Hay dos métodos que son los más utilizados para la generación. Ambos se basan en la siguiente definición:
# 
# **Definición.** Dos enteros $a$ y $b$ son congruentes módulo $m$ si su diferencia es un múltiplo entero de $m$ y se expresa como
# $$𝑎 \equiv 𝑏 ( \text{𝑚o𝑑 } 𝑚)$$
# Como consecuencia:
# - $(a-b)$ es divisible entre $m$
# - $a$ y $b$ dan el mismo residuo al ser divididos entre $m$

# #### Método congruencial multiplicativo
# 
# Generar una secuencia de números pseudoaleatorios uniformes en el intervalo (0, 1) usando la siguiente fórmula recursiva:
# 
# $$n_{i+1} = a n_i \mod m$$
# - $m$ debe ser tan grande como sea posible, dependiendo de los bits por palabra que maneje la computadora, descontando el bit del signo ($b$). Por lo tanto: $m = 2^b$
# - $a$ debe satisfacer que $a \approx 2^{(b+1)/2}$ y que $a \equiv \pm 3 \mod 8$. La segunda expresión equivale a $( a-(\pm3 ) )$ es múltiplo de $8$.
# - $n_0$: entero positivo impar menor a $m$
# - El periodo será de longitud: $m/4$

# In[39]:


# n0 : valor inicial o semilla
# a : multiplicador
# m : modulo (2^b)
# n : el número de numeros que quiero generar

def multiplicativo(n0,a,m,n):
    secuencia = []
    ni = n0 # iniciar con una semilla
    for _ in range(n):
        ni = (a * ni) % m # calculo el sig numero utilizando la formula recurrente
        secuencia.append(ni/m) # normaliza entre 0 y 1
    return secuencia


# In[40]:


m = 2**31
a = 65539 # a ≡  +-3 mod 8
n0 = 12345
n=10

secuencia_mult = multiplicativo(n0,a,m,n)
print(secuencia_mult)


# In[41]:


import numpy as np
import plotly.graph_objects as go # Para gráficos interactivos

# Generamos ambas secuencias
secuencia_mult = multiplicativo(n0,a,m,n)
secuencia_numpy = np.random.uniform(0,1,n)

# figura con Plotly
fig = go.Figure()

# Añadir los histogramas
fig.add_trace(go.Histogram(
    x=secuencia_mult,
    nbinsx = 40,
    opacity = 0.6,
    name = 'Método multiplicativo',
    marker_color='red'
))

fig.add_trace(go.Histogram(
    x=secuencia_numpy,
    nbinsx = 40,
    opacity = 0.6,
    name = 'Numpy random uniform',
    marker_color='blue'
))

fig.update_layout(
    barmode='overlay', #superpone los histogramas
    title = 'Compararación: Método multiplicativo -vs- Numpy',
    xaxis_title='valor generado',
    yaxis_title='Frecuencia',
    legend_title='Método',
    bargap=0.05
)

fig.show()


# #### Método congruencial mixto
# 
# Este método genera números pseudoaleatorios con la fórmula: 
# $$n_{i+1} = (a n_i + c) \mod m$$
# 
# **Obs.** Se le llama “mixto” porque incluye una constante adicional $c$ (a diferencia del método multiplicativo).
# 
# - $m = 2^b$
# - $a \approx 2^{(b-1)/2}$, $a \equiv 1 \mod 4$
# - $c$, $n_0$: enteros positivos impares $< m$
# - Periodo: $m$

# In[42]:


# n0 : valor inicial o semilla
# a : multiplicador
# c : deber ser impar
# m : modulo (2^b)
# n : el número de numeros que quiero generar

def mixto(n0,a,c,m,n):
    secuencia = []
    ni = n0 # iniciar con una semilla
    for _ in range(n):
        ni = (a * ni + c) % m # calculo el sig numero utilizando la formula recurrente
        secuencia.append(ni/m) # normaliza entre 0 y 1
    return secuencia


# In[43]:


m = 2**31
a = 1103515245 # a cong 1 mod----
c = 12345
n0 = 42
n= 10

secuencia_mixto = mixto(n0,a,c,m,n)
print(secuencia_mixto)


# In[44]:


# Generamos ambas secuencias
secuencia_mixto = mixto(n0,a,c,m,n)
secuencia_numpy = np.random.uniform(0,1,n)

# figura con Plotly
fig = go.Figure()

# Añadir los histogramas
fig.add_trace(go.Histogram(
    x=secuencia_mixto,
    nbinsx = 40,
    opacity = 0.6,
    name = 'Método mixto',
    marker_color='red'
))

fig.add_trace(go.Histogram(
    x=secuencia_numpy,
    nbinsx = 40,
    opacity = 0.6,
    name = 'Numpy random uniform',
    marker_color='blue'
))

fig.update_layout(
    barmode='overlay', #superpone los histogramas
    title = 'Compararación: Método mixto -vs- Numpy',
    xaxis_title='valor generado',
    yaxis_title='Frecuencia',
    legend_title='Método',
    bargap=0.05
)

fig.show()


# In[45]:


import random
n = 1000

# Generar secuencia usando random.random()
secuencia_random = [random.random() for _ in range(n)]

# Generar secuencia usando numpy
secuencia_numpy = np.random.uniform(0, 1, n)

# Graficar histogramas
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=secuencia_random,
    nbinsx=40,
    opacity=0.6,
    name='random.random()',
    marker_color='crimson'
))

fig.add_trace(go.Histogram(
    x=secuencia_numpy,
    nbinsx=40,
    opacity=0.6,
    name='numpy.random.uniform',
    marker_color='royalblue'
))

fig.update_layout(
    barmode='overlay',
    title='Comparación: random.random() vs numpy.random.uniform',
    xaxis_title='Valor generado',
    yaxis_title='Frecuencia',
    legend_title='Método',
    bargap=0.05
)

fig.show()


# ### Metodos de generación de numeros aleatorios no rectangulares

# #### Método de la transformada inversa
# 
# El método utiliza la función de distribución $F(x)$ de la distribución que se va a simular 
# $$F(x) = \int_{-\infty}^{x} f(t)\,dt$$

# Como se sabe, los valores de $F(x)$ están en el intervalo $(0,1)$ al igual que los números rectangulares $U$. 
# 
# El método genera un $U$ y trata de determinar el valor de la variable aleatoria para la cual $F(x)$ sea igual a $U$.
# 
# Si $U \in (0,1)$:
# $$ F(x) = U \quad \Rightarrow \quad x = F^{-1}(U)$$

# $\color{slateblue}{\text{TEOREMA.}}$ Sea $X$ una variable aleatoria real. Supongamos que su función de distribución $F$ es estrictamente creciente (por lo que $F$ es una biyección de $\mathbb{R}$ sobre $(0,1)$ y podemos denotar por $F^{-1}$ a su inversa). Sea $U\sim \text{unif}[0,1]$. Entonces $F^{-1}(U)$ tiene la misma ley que $X$.
# 
# Si $F$ no es estrictamente creciente, hemos visto que el teorema precedente sigue siendo válido bajo la condición de definir 
# $$F^{-1}(u)=\inf\{ x\in \mathbb{R} : F(x)\geq u\},$$
# la inversa generalizada de $F$.

# In[46]:


# Vamos a simular una v.a. Bernoulli usando una Uniforme
import random
def bernoulli(p):
    u = random.random() # U -> Unif(0,1)
    return 1 if u <= p else 0


# In[47]:


# Simulamos n valores con p de éxito
p = 0.5
simulaciones = [bernoulli(p) for _ in range(100)]
print(simulaciones)


# In[48]:


sum(simulaciones)


# ### Ejemplo: Distribución exponencial
# 
# sea $\theta,\lambda>0$, entonces a través del teorema se puede generar una v.a. exponencial.
# 
# Sea $U \sim Unif[0,1]$, entonces si $X\sim Exp(\lambda)$, entonces
# 
# $$f_{X}(x) = \lambda e^{-\lambda x}$$ 
# y que 
# $$ F_X(x) = 1-e^{-\lambda x} $$
# 
# Sabemos que $1-e^{-\lambda x} = U$, entonces $1-U = e^{-\lambda x}$, y tomando logaritmo, se tiene que 
# $ \ln(1-U) = -\lambda x$ y como $U$ es uniforme entonces $U \sim 1-U$, por lo que depejando a $x$, se tiene que 
# $$ X = -\frac{\ln(U)}{\lambda} \sim Exp(\lambda) $$

# ### Ejemplo: Distribución Uniforme Continua
# $$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$
# $$F(x) = \frac{x-a}{b-a} = U \Rightarrow x = a + (b-a)U$$

# In[49]:


import numpy as np
import plotly.graph_objects as go

# Semilla
np.random.seed(123)

# Parámetro de la exponencial
lambd = 4
n = 10000

# Simulación usando transformada inversa
uniformes = np.random.random(n)
exponenciales = -np.log(uniformes) / lambd

# Curva teórica
x_vals = np.linspace(0, exponenciales.max(), 300)
y_vals = lambd * np.exp(-lambd * x_vals)

# Gráfico
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=exponenciales,
    nbinsx=40,
    histnorm='probability density',
    marker_color='skyblue',
    name='Simulación'
))

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    name='Densidad teórica',
    line=dict(color='darkblue')
))

fig.update_layout(
    title='Distribución Exponencial simulada vs teórica (λ = 1)',
    xaxis_title='x',
    yaxis_title='Densidad',
    bargap=0.05
)

fig.show()


# In[50]:


# Parámetros de la uniforme
a, b = 2, 5
n = 10000

# Simulación por transformación lineal
uniformes = np.random.random(n)
uniforme_continua = a + (b - a) * uniformes

# Densidad teórica (constante)
x_vals = np.linspace(a, b, 300)
y_vals = np.ones_like(x_vals) * (1 / (b - a))

# Gráfico
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=uniforme_continua,
    nbinsx=40,
    histnorm='probability density',
    marker_color='orange',
    name='Simulación'
))

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    name='Densidad teórica',
    line=dict(color='red')
))

fig.update_layout(
    title='Distribución Uniforme Continua simulada vs teórica [2, 5]',
    xaxis_title='x',
    yaxis_title='Densidad',
    bargap=0.05
)

fig.show()


# $\color{indigo}{\text{EJEMPLO.}}$ (Variable aleatoria Cauchy) Simulemos la variable aleatoria de Cauchy de parámetro $1$ que tiene por función de densidad 
# $$\frac{1}{\pi}\frac{1}{1+x^{2}}.$$
# 
# $\color{purple}{\text{EJERCICIO.}}$
# - Corroborar que la función así definida es una función de densidad de probabilidad
# - Encontrar $F_X$
# - Crear una gráfica en Python que muestre su forma.
# 
# Por el teorema anterior, para $u\in (0,1)$
# $$u=\frac{1}{\pi}\arctan(x)+\frac{1}{2} \qquad \text{si y sólo si} \qquad x=\tan\left({\pi}\left(u-\frac{1}{2} \right) \right).$$

# - Corroborar que la función así definida es una función de densidad de probabilidad
# 
# La función de densidad propuesta es:
# 
# $$f_X(x) = \frac{1}{\pi (1 + x^2)}$$
# 
# Se verifica que:
# 
# - $$ f_X(x) \geq 0 \forall x \in \mathbb{R} $$
# - $$ \int_{-\infty}^{\infty} f_X(x) \, dx = 1 $$
# 
# Sabemos que:
# 
# $$
# \int_{-\infty}^{\infty} \frac{1}{1 + x^2} dx = \pi
# \quad \Rightarrow \quad
# \int_{-\infty}^{\infty} f_X(x) dx = \frac{1}{\pi} \cdot \pi = 1
# $$
# 
# Por lo tanto, es una función de densidad de probabilidad válida.
# 

# - Encontrar $F_X$
# 
# Para la variable Cauchy(1), la función de distribución acumulada $F_X$ es:
# 
# $$F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt = \frac{1}{\pi} \arctan(x) + \frac{1}{2}$$
# 

# - Crear una gráfica en Python que muestre su forma.

# In[57]:


# Valores de x
x = np.linspace(-10, 10, 1000)

# Función de densidad
f_x = 1 / (np.pi * (1 + x**2))

# Función de distribución acumulada
F_x = (1/np.pi) * np.arctan(x) + 0.5

# Gráficas
plt.figure(figsize=(12, 5))

# Gráfico de la densidad
plt.subplot(1, 2, 1)
plt.plot(x, f_x, label='f(x)', color='blue')
plt.title('Función de densidad de Cauchy(1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Gráfico de la acumulada
plt.subplot(1, 2, 2)
plt.plot(x, F_x, label='F(x)', color='green')
plt.title('Función de distribución acumulada')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Por el teorema anterior, para $u\in (0,1)$
# $$u=\frac{1}{\pi}\arctan(x)+\frac{1}{2} \qquad \text{si y sólo si} \qquad x=\tan\left({\pi}\left(u-\frac{1}{2} \right) \right).$$
# 
# Utilizando la fórmula de la inversa de la función de distribución acumulada:
# 
# $$
# F_X(x) = \frac{1}{\pi} \arctan(x) + \frac{1}{2} \quad \Longrightarrow \quad
# x = \tan\left( \pi (u - \frac{1}{2}) \right)
# $$
# 
# Para simular una variable Cauchy, generamos valores $ u \sim \text{Uniform}(0, 1)$, y aplicamos la transformación.
# 

# In[58]:


# Simulación
u = np.random.uniform(0, 1, 10000)
x_sim = np.tan(np.pi * (u - 0.5))

# Histograma de valores simulados
plt.hist(x_sim, bins=200, density=True, alpha=0.6, color='gray', label='Simulación')

# Densidad teórica
x_vals = np.linspace(-10, 10, 1000)
f_x_vals = 1 / (np.pi * (1 + x_vals**2))
plt.plot(x_vals, f_x_vals, color='red', label='Densidad teórica')

plt.title('Simulación de variable aleatoria Cauchy(1)')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.xlim([-10, 10])
plt.show()


# $\color{purple}{\text{EJERCICIO.}}$
# 
# Se tienen $2$ urnas, cada una de las cuales contiene $10$ bolas numeradas del $1$ al $10$. 
# 
# Un experimento aleatorio consiste en seleccionar al azar una bola de cada urna. 
# 
# Sea $X$ la variable aleatoria que representa la diferencia de los números de las dos bolas seleccionadas.
# 
# * Define $\Omega$
# * Calcula las probabilidades asociadas
# * Define la función de densidad
# * Verifica que $\sum_{x\in R_X}f_X(x) = 1$
# * Gráfica la función de densidad
# 
# Además, calcula la probabilidad de que los números de las dos bolas seleccionadas difieran por $2$ o más.

# In[59]:


# Definición del espacio muestral 
from itertools import product 
from fractions import Fraction 
from collections import defaultdict
import pandas as pd
import numpy as np

#Espacio muestral
espacio_muestral = set(product([1,2,3,4,5,6,7,8,9,10], repeat=2))

#Cardinalidad
cardinalidad= len (espacio_muestral)


#Cálculo de diferencias
def S(n):
    Sn={(i,j) for i in range (1,11) for j in range (1,11) if abs(i-j)==n}
    return Sn

#Calculo de probabilidades 
def P(A):
    P=Fraction(len(A), len (espacio_muestral))
    return P

#Diccionario 
S={(i,j): abs(i-j) for i,j in espacio_muestral}

dS= defaultdict(set)

for (i,j) in S.items(): 
    dS[j].add(i)

list (dS.items())

leyS={i: P(A) for i, A in dS.items()}
ley_S= pd.Series(leyS)
ley_S.sort_index()

#Función de densidad 
def leyS(i):
    if i in range (0,10):
        x=ley_S[i]
    else:
        x=0 # Si no esta en el rango me devuelve 0
    return x

rango_S=[k for k in range(0,10)]
p_k=[float(leyS(k)) for k in range (0,10)] 
lS= pd.DataFrame(list(zip(rango_S, p_k)), index=[f"S_{i}" for i in range (1,11)], columns= ['rango', 'densidad'] )

Total= .10+.18+.16+.14+.12+.10+.08+.06+.04+.02

print ('Espacio muestral=', espacio_muestral)
print ('\nCardinalidad=', cardinalidad)
print ('\nPorbabilidades=', ley_S.sort_index())
print ('\nFunción de densidad', lS)
print ('\nAcumulada', Total)
lS.plot.bar('rango', 'densidad')
print ('\nProbabilidad    X>-2', 1-.1-.18)


# ## Teorema Central del Límite (TCL)
# 
# Cuando vemos una v.a. binomial que es muy grande, se dice que converge a Poisson, esto debido a que, cuando n es muy grande en lugar de calcular la probabilidad se hace una estandarización para poder aproximar los valores de una v.a. discreta a una contínua, por lo que $E[X]=np$ y $Var(x)=np(1-p)$ de manera que, la estandarización es la siguiente:
# 
# $$Z = \frac{x-E[x]}{\sqrt{Var(x)}}\sim N(0,1)$$
# 
# Un vector aleatorio es una función $(x,y):\Omega \to \mathbb{R^2}$ donde $x$ e $y$ son v.a.
# 
# Sean $X_1,X_2,\dots,X_n$ y $Y_1,Y_2,\dots,Y_n$ v.a., con $X$ e $Y$ independientes, la distribución de probabilidad conjunta se define como
# 
# $$P(X=x, Y=y) = f_{x,y}(x,y) = f_x(x)f_y(y)$$
# 
# $$\sum_{x}\sum_{y}f_{x,y}(x,y)=1$$
# 
# La función de densidad conjunta contiene información de ambas variables aleatorias.
# 
# A partir de la función de densidad conjunta es posible calcular las probabilidades marginales.
# 
# Probabilidad Marginal de X
# 
# $$f_x(x)=\sum_{y \in R_y}f_{x,y}(x,y)$$
# 
# Probabilidad Marginal de Y
# 
# $$f_y(y)=\sum_{x \in R_x}f_{x,y}(x,y)$$
# 
# Son funciones de densidad que cumplen con las siguientes características:
# - X y Y son v.a. independientes.
# - Son identicamente distribuídas.
# - $f_{x,y}(x,y)\geq0$
# 
# Por lo que se tiene lo siguiente:
# 
# $$f_{x,y}(x,y)\geq0$$
# 
# $$\int\int{f_{x,y}(x,y)dxdy}=1$$
# 
# $$f_x(x)=\int{f_{x,y}(x,y)dy}$$
# 
# $$f_y(y)=\int{f_{x,y}(x,y)dx}$$
# 
# Decimos que $X_1,X_2,\dots,X_N$ son v.a.i si $f_x(x_1,\dots,x_n)=f_{x_1}(x_1),\dots,f_{x_n}(x_n)$
# 
# Si son identicamente distribuídas, se refiere a que $X_1,X_2,\dots,X_N\sim N(0,1)$, por lo que todas las $x_i$ tienen la misma densidad, de manera que 
# 
# $$E[x_i]=\mu \in R$$
# 
# $$Var(x_i)=\sigma^2>0$$
# 
# El TLC y la LGN asumen que la varianza es finita, de manera que $E[x_i]<\infty$ y $Var(x_i)<\infty$
# 
# Me gustaría calcular el límite de dos variables finitas 
# 
# $S_n=x_1+x_2+\dots,x_n$$
# 
# A partir de esto, queremos calcular la esperanza de $S_n$ 
# 
# $$E[S_n]=E\left[\sum_{i=1}^n{x_i}\right]=\sum_{i=1}^n{E[x_i]}=n\mu$$
# 
# Entonces,
# 
# $$Var(S_n)=Var\left(\sum_{i=1}^n{x_i}\right)=\sum_{i=1}^n{Var(x_i)}=n\sigma^2$$
# 
# Esto gracias a que son independientes.
# 
# Recordemos que la varianza nos indica qué tan lejanos son los valores entre sí y la covarianza nos indica qué tan alejados están los valores de la media.
# 
# $\color{indigo}{\text{NOTA.}}$ Si no son independientes no es cierto que la varianza sea lineal.
# 
# $$+2\sum_{i<j}cov(x_i,y_j)$$
# 
# $$Cov(x,y) = E\left[ (x-E[x]) (y-E[y]) \right]$$
# 
# La $Cov(x,y)$ es una medida de dispersión conjunta.
# 
# Ahora entonces, puedo hacer una estimación
# 
# $$Z_n = \frac{S_n - n\mu}{\sqrt{n\sigma^2}} \sim N(0,1)$$
# 
# $$Z_n \mapsto Z \sim N(0,1)$$
# 
# $x_1,\dots,x_n \mapsto X$ por lo que $F_{x_1},\dots,F_{x_n}(x_n)$
# 
# De manera que,
# $$P(a \leq S_n \leq b) = \phi(b) - \phi (a)$$
# 
# Por lo que,
# $$\phi (Z) = P(Z \leq z) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}}e^{-x^2/2}dx$$
# 
# # La Ley de los Grandes Números (LGN)
# 
# La Ley de los Grandes Números dice que todo converge al valor esperado, es decir, si me tomo un $M_n$, que es un promedio aritmético, se puede decir que $M_n$ converge al valor esperado de $x_i$ es decir:
# 
# $$M_n = \frac{S_n}{n} \implies M_n \mapsto E[x_i] = \mu, \forall  {n\to\infty}$$
# 
# Es decir, si tengo una v.a.i. $x_1+,\dots,+x_n\sim Bernoulli(p)$
# 
# $$\frac{x_1+,\dots,+x_n}{n}= \mu =p$$
# 
# De igual manera, si tengo una v.a.i. $x_1+,\dots,+x_n\sim Unif(0,1)$
# 
# $$\frac{x_1+,\dots,+x_n}{n} = \frac{1}{2}= \mu$$
# 
# En general, se puede decir que sirve para garantizar una estabilidad en promedios en el largo plazo ya que n es muy grande, dependiendo del tipo de convergencia.
# 
# $$\text{Ley débil} \mapsto \text{en P}$$
# 
# 

# ### Aplicación del TLC y TGN

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Simulación de la LGN

# Simularemos los lanzamientos de un dado y calcularemos la media.
# 
# $$ \Omega = \{1,2,3,4,5,6\}$$
# 
# La media teórica: $$\mathbb{E}(X) = 3.5$$

# In[54]:


np.random.seed(42) # Para garantizar que los números pseudoaleatorios sean reproducibles
lanzamientos = np.random.randint(1,7,10000)
lanzamientos


# In[55]:


media_acum = np.cumsum(lanzamientos) / np.arange(1,10001)
df = pd.DataFrame({
    'Lanzamiento': np.arange(1,10001),
    'Media Acumulada': media_acum})


# In[56]:


plt.figure(figsize=(10,6))
plt.plot(df['Lanzamiento'],df['Media Acumulada'],label='Media Muestral')
plt.axhline(3.5, color='red',linestyle='--',label='Media Teórica (3.5)')
plt.xlabel('Número de lanzamientos')
plt.title('LGN en lanzamientos de un dado')
plt.legend()
plt.show()

