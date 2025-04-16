#!/usr/bin/env python
# coding: utf-8

# # Estadística

# ## Estimadores
# 
# En estadística se tiene una población de $N$ individuos la cual tiene parámetros $\mu$ y $\sigma^2$, que son la media y la varianza poblacionales respectivamente. Como es complicado estudiar una población completa, se suele tomar muestras aleatorias representativas de lo que queremos estudiar.
# 
# De acuerdo a esto, en lugar de tener parámetros se tiene estimadores, por eso se dice que se puede estimar a través de un conjunto de datos con lo cual se puede estimar la media y la varianza muestrales, $\bar{X}$ y $\delta^2$ respectivamente.
# 
# Sea $(X_1,\dots,x_n)$ una m.a.
# 
# $$\bar{X} = \frac{\sum{x_i}}{n}$$
# $$\delta^2 = \frac{\sum(x_i-\bar{x})^2}{n-1}$$
# 
# Se quiere conocer el "mejor parámetro" dependiendo del conjunto de datos.
# 
# Estimadores: $T(x_1,\dots,x_n)$
# 
# $$T(x_1,\dots,x_2) = \frac{x_i+x_2}{n}$$
# 
# $$T(x_1,\dots,x_n) = \frac{x_1,\dots,x_n}{n}$$
# 
# $$\phi = (\theta, \theta \text{es parámetro de la distribución})  \underline{C} \mathbb{R}$$
# 
# De manera que los estimadores solo dependen de los valores muestrales y no de los poblacionales.
# 
# $$E[x_i], Var(x_i) < \infty$$
# 
# Es decir, son estimadores que además de ser finitos, cumplen con las siguientes características:
# - Insesgados
# - Consistentes
# - Completos
# - Suficientes
# 
# Esto aplica para los estimadores
# - Máxima verosimilitud
# - Mínimos cuadrados
# - Método de momentos
# 
# Sea $(x_1,\dots,x_n)$ v.a.i. cada una con $f_{x_i}(x_i)$.
# 
# El conjunto de v.a. $(x_1,\dots,x_n)$ se denomina una m.a. de tamaño $n$ cualquier función real de una m.a. $T(x_1,\dots,x_n)$ se llama **estadístico**.
# 
# Sea $X$ una v.a. con función de densidad $f(x;\theta)$ que indica que depende de los parámetros desconocidos $x$ y $\theta$.
# 
# Sea $(x_1,\dots,x_n)$ una m.a. de x, se tiene
# $$f(x;\theta) = f(x_1,\dots,x_n) = \prod_{i=1}^n f(x;\theta)$$
# 
# $$f_{x_1,\dots,x_n}(x_1,\dots,x_n) = (f_{x_1}(x_1))\dots(f_{x_n}(x_n))$$
# 
# donde $x_i$ son los valores de los datos.
# 
# Un estimador del parámetro $\theta$ es cualquier estadístico $T(x_1,\dots,x_n)$ t se denota por
# 
# $$\hat\theta = T(x_1,\dots,x_n)$$
# 
# $\color{slateblue}{\text{NOTA.}}$ Un estimador es una v.a.
# 
# Si los estimadores están en función de un solo valor, entonces se tiene un estimador puntual.
# 
# Estimadores:
# - Puntuales: En función de un solo valor.
# - Intervalo: En función de un rango de valores.

# ### Distribuciones Muestrales Relacionadas con la Distribución Normal
# 
# Sean $Y_1,...,Y_n$ v.a. independientes $\sim N \left(\mu,\frac{\sigma^2}{n}\right)$
# 
# $$\implies \bar{Y} = \frac{1}{n} \sum_{i=1}^n y_i \sim N \left( \mu,\frac{\sigma^2}{n}\right)$$
# 
# Hay otras distribuciones que me interesa estudiar porque están relacionadas con la estadística inferencial.
# 
# Estandarizando se tiene
# $$Z_i = \frac{Y_i-\mu}{\sqrt{\sigma^2}}=\frac{Y_i-\mu}{\sigma} \sim N(0,1)$$
# 
# Si tomamos 
# $$\sum_{i=1}^n {Z_i}^2 = \sum_{i=1}^n\left(\frac{Y_i - \mu}{\sigma}\right)^2 \sim {X_n}^2$$
# 
# * Se distribuye $X^2$ con $n$ grados de libertad
# 
# $$\implies {Z_i}^2 \sim {X_n}^2$$
# 
# La $X^2$ es importante en la estadística inferencial cuando se quiera hacer inferencia acerca de la varianza poblacional.
# 
# Recordando que los parámetros poblacionales son $\mu$ y $\sigma^2$.
# 
# Si yo quisiera hacer inferencia acerca de la media de una población normal $\mu$ con varianza desconocida, se puede estimar.
# 
# Cuando no conocemos $\sigma$ se estima con $\delta = \sqrt{\delta^2}$
# 
# Entonces se busca desarrollar métodos de inferencia respecto de $\mu$.
# 
# Si $Z \sim N(0,1)$ y $W \sim {X_n}^2$, con $Z$ y $W$ independientes, entonces
# $$T=\frac{Z}{\sqrt{W/n}} = \sqrt{n}\left(\frac{\bar{Y}-\mu}{\delta}\right) \sim t_{n-1}$$
# 
# * Se distribuye $t$ de student con n grados de libertad
# 
# Si $W_1$ Y $W_2$ son v.a.i. con $w_1 \sim {X_{n_1}}^2$ y $w_2 \sim {X_{n_2}}^2$
# 
# \begin{align*}
# \mathbb{F}& = \frac{w_1/n_1}{w_2/n_2} \\
#           & = \frac{{\delta_1}^2/{\sigma_1}^2}{{\delta_2}^2/{\sigma_2}^2} 
# \end{align*}

# ### Propiedades de los Estimadores
# 
# Sea $(x_1,\dots,x_n)$ una m.a. que proviene de una v.a. con función de densidad $f_x(x)$ donde $\mathbb{E}[x]=\mu$ y $Var(x)=\sigma^2$
# 
# $$T(x) = \bar{X} = \frac{x_1,\dots,x_n}{n}$$
# 
# El estimador es insesgado si $\mathbb{E}[\bar{X}]=\mu$
# 
# De manera más general, un estimador $\hat\theta = T(x_1,\dots,x_n)$ es insesgado de $\theta$ si su esperanza $\mathbb{E}[\hat\theta]=\theta$.
# 
# Para los estimadores insesgados había un error cuadrático medio.
# 
# Si $\hat\theta$ es insesgada, entonces su error cuadrático medio es
# $$\mathbb{E}[(\hat\theta - \theta)] = Var(\theta)$$
# 
# Recordando que la FGM nos sirve para encontrar los momentos de cualquier orden.
# 
# Si $X$ es una v.a. con esperanza $\mu$ y $n \in \mathbb{N}$, con $\mathbb{x^n}$ el valor esperado en el n-ésimo momento de x,
# 
# * POBLACIONALES
# $$M_x(t) → M^{(n)}(0)=\mathbb{E}[x^n]$$
# 
# El método de momentos busca igualar los momentos muestrales con los momentos poblacionales
# 
# * MUESTRALES
# $$m_n = \frac{1}{n}\sum_{i=1}^k {x_i}^n$$
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Hallar los estimadores por el método de momentos para $(x_1,\dots,x_n)$ una m.a. de una población con densidad $Gamma(\alpha,\beta)$
# 
# Encontrar estimadores para $\alpha$ y $\beta$ $i.e.$ $\hat\alpha$ y $\hat\beta$
# 
# Si $X \sim \gamma(\alpha,\beta)$
# $$\implies \mathbb{E}[x]=\frac{\alpha}{\beta}$$
# $$ \implies Var(x)\frac{\alpha}{\beta^2}$$
# 
# Si sé que 
# $$Var(x)=\mathbb{E}[x^2]-(\mathbb{E}[x]^2)$$
# 
# \begin{align*}
# \implies\mathbb{E}[x^2] & = Var(x)+(\mathbb{E}[x])^2 \\
#                         & = \frac{\alpha}{\beta^2} + \frac{\alpha^2}{\beta^2}
# \end{align*}
# 
# Tengo que igualar el momento poblacional con el muestral y tengo que igualar el segundo momento poblacional con el muestral.
# 
# $$\mathbb{E}[x]=\frac{1}{n}\sum{x_i}$$
# 
# $$\mathbb{E}[x^2] = \frac{1}{n}\sum{x_i}^2$$
#  
# $$\implies \frac{\alpha}{\beta}=\frac{1}{n}\sum{x_i}$$
# 
# $$\implies \frac{\alpha}{\beta}=\bar{X}$$
# 
# $$\implies \beta=\frac{\alpha}{\bar{X}}$$
# 
# Sustituyendo
# $$\frac{\alpha}{\beta^2} + \frac{\alpha^2}{\beta^2} = \frac{1}{n}\sum{x_i}^2$$
# 
# Finalmente, tenemos que
# 
# $$\hat\alpha = \frac{n\bar{x^2}}{\sum{(x_i - \bar{x})^2}}$$
# 
# $$\hat\beta = \frac{n\bar{x}}{\sum{(x_i - \bar{x})^2}}$$
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Sea $X_i=(x_1,\dots,x_n)$ una m.a.
# 
# $$f(x;\theta)=\theta(1+x)^{-(1+\theta)}, \forall x \in (0,\infty), \theta > 0$$
# 
# Estimar $\theta$ por momentos asumiendo que $\theta>1$
# 
# $$\mathbb{E}[x]=\frac{\partial{M_x(t)}}{\partial t}$$
# 
# $X_i \sim Bernoulli(p)$ v.a.i.i.d.
# 
# \begin{align*}
# L(X_i;p) & = (\mathbb{P}(X=x_1;p))(\mathbb{P}(X=x_2;p))\dots (\mathbb{P}(X=x_n;p)) \\
#                    & = (p^{x_1}(1-p)^{1-x_1})(p^{x_2}(1-p)^{1-x_2})\dots(p^{x_n}(1-p)^{1-x_n}) \\
#                    & = p^{\sum{x_i}}(1-p)^{n-\sum{x_i}}
# \end{align*}
# 
# \begin{align*}
# \implies ln[L(X_i;p)] & = ln\left(p^{\sum{x_1}}(1-p)^{n-\sum{x_i}}\right) \\
#                               & = \sum{x_iln(p)}+\left(n-\sum{x_i}\right)ln(1-p)
# \end{align*}
# 
# $$\implies \frac{\partial{ln[L(X_i;p)]}}{\partial p} =\frac{1}{p}\sum{x_i}+\left(n-\sum{x_i}\right)\left(\frac{1}{1-p}(-1)\right)= 0$$
# 
# $$\implies\frac{(1-p)\sum{x_i}-p(n-\sum{x_i})}{p(1-p)}$$
# 
# $$p=\frac{1}{n}\sum{x_i}$$
# 
# $$\hat{p} = \frac{\sum{x_i}}{n} \rightarrow p$$
# 
# * Converge al p real.

# ## Estimador Máximo Verosímil

# Existen distintos métodos para estimar parámetros desconocidos a partir de un conjunto de datos. El estimador máximo verosímil (MLE) responde a la siguiente pregunta:
# ¿Para qué valor del parámetro, los datos observados tienen la probabilidad más alta?

# **Funcion de verosimilitud**
# Sea $X_1,X_2,\dots,X_N$ una m.a. con función de densidad $f(x;\theta)$. La función de verosimilitud (likelihood) es:
# 
# *Como es una muestra independiente sabemos que va desde $x_1$ hasta $x_n$*
# 
# $$ f(x_1,\dots,x_n;\theta) = f(x_1;\theta)\cdots f(x_n;\theta) $$
# 
# Entonces,
# 
# $$ L(x_1,\dots,x_n;\theta) = \prod_{i=1}^n f(x_i,\theta) $$
# 
# Ejemplo. Si $X_1,\dots , X_n$ es una m.a. Poisson($\lambda$), entonces,
# $$ L(x_1,\dots,x_n;\lambda) = \frac{e^{-n\lambda} \lambda ^{\sum x_i}}{\prod x_i !} $$

# **Estimador Máximo Verosímil** es el valor de $\theta$, donde $L(x_1,\dots,x_n;\theta)$ alcanza el máximo.
# 
# Pasos:
# - $$ L(x_1,\dots,x_n;\theta) = \prod_{i=1}^n f(x_i,\theta) $$
# - $$\ln L(x_1,\dots,x_n;\theta) $$
# - $$ \frac{d\ln L(x_1,\dots,x_n;\theta)}{d\theta} $$
# - $$ \frac{d\ln L(x_1,\dots,x_n;\theta)}{d\theta} = 0 $$
#  
# y despejar $\theta$ para encontrar el máximo.

# $\color{indigo}{\text{EJEMPLO.}}$
# Una moneda se lanza 100 veces. Dado que se obtienen 55 soles, queremos encontrar el MLE para la probabilidad $p$ de obtener sol en un solo lanzamiento.
# 
# Podemos pensar en contar el número de soles en los 100 lanzamientos, entonces la probabilidad de obtener 55 soles en este experimento, es la siguiente:
# 
# Si $X$: el número de soles que se obtiene al lanzar una moneda, se tiene que $X\sim Bin(100,p)$, es decir, 
# 
# (*Como es una binomial y se quiere obtener los parámetros, entonces se define así:*)
# $$ f(55 soles; p) = \binom{100}{55}p^{55}(1-p)^{100-55} $$
# 
# Si yo quiero estimar este parámetro p que desconozco tengo que aplicar la función de verosimilitud.
# 
# Para nuestro caso, tenemos
# $$ \frac{P(55 soles ;p)}{dp}=\binom{100}{55}p^{55}(1-p)^{45} - 45p^{55}(1-p)^{44}=0$$
# entonces, resolviendo para $p$, se tiene
# $$ \binom{100}{55}p^{55}(1-p)^{45}=45^{55}(1-p)^{44} $$
# entonces,
# $$ 55(1-p) = 45p $$
# entonces, 
# $$p = \frac{11}{20}$$
# 
# Por lo tanto, el MLE para $p$, es $\hat{p}=0.55$

# In[1]:


from scipy.stats import bernoulli, binom
import numpy as np
import sympy # para cálculo simbólico y algebráico -> encontrar el estimador de manera analítica
from sympy.abc import x

#Definir a p como una variable simbólica positiva.

p = sympy.symbols('p',positive=True) # Porque representa una posibilidad
f = p**55 * (1-p)**(100-55) # Esta es nuestras función de verosimilitud
# Lo que quiero es \hat{p}
phat=sympy.solve(sympy.diff(f,p),p)[0]
print("El estimador de máxima verosimilitud es",phat)


# Veamos otro ejemplo:

# In[2]:


r= binom.rvs(1,0.7,size=100)
r


# In[3]:


f = p**x * (1-p)**(1-x)
J = np.prod([f.subs(x,i) for i in r])
logJ = sympy.expand_log(sympy.log(J))
phat = sympy.solve(sympy.diff(logJ,p),p)[0]
print("El estimador de máxima verosimilitud es",phat)


# $\color{purple}{\text{EJERCICIO 1.}}$ El tiempo de retardo para pacientes con enfermedad coronaria potencial se define como la duración entre el comienzo de un síntoma agudo cardiovascular y la llegada a la unidad de emergencia.
# 
# Se supone que el tiempo de retardo sigue una distribución exponencial con parámetro $\theta $.
# 
# Se registraron los tiempos de retardo (en minutos) en la clínica para los primeros 20 pacientes:
# 
# $$ 525, \ 719, \ 2880, \ 150, \ 30, \ 251, \ 45, \ 858, \ 15, \ 47, \ 90, \ 56, \ 68, \ 6, \ 189, \ 180, \ 60, \ 60, \ 294, \ 747 $$
# 
# - Encuentra un estimador por el método de momentos para la media de la distribución.
# - Encuentra el valor del estimador con los datos dados.

# $\color{purple}{\text{SOLUCIÓN}}$
# 
# Considerando una distribución exponencial con parametro $\theta$, sabemos que por definición la media teórica es:
# 
# $$E[X] = \theta$$
# 
# Tomando en cuenta el *método de momentos*,  igualamos la media teórica con la media muestral, dicho de otra forma.
# 
# $$\hat{\theta} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$$
# 
# Dado nuestro conjunto de datos:
# 
# $$525, 719, 2880, 150, 30, 251, 45, 858, 15, 47, 90, 56, 68, 6, 189, 180, 60, 60, 294, 747$$
# 
# A su vez 
# $$n=20$$
# 
# La media muestral (media empirica) es la siguiente:
# 
# $$\bar{X} = \frac{525 + 719 + 2880 + 150 + 30 + 251 + 45 + 858 + 15 + 47 + 90 + 56 + 68 + 6 + 189 + 180 + 60 + 60 + 294 + 747}{20}$$  
# 
# \begin{align*}
# \bar{X} & = \frac{7270}{20} \\
#         & = 363.5
# \end{align*}
# 
# Por lo tanto, nuestro estimador *$\theta$* calculado por el método de momentos es:
# 
# *$$\hat{\theta} = 363.5$$*

# $\color{purple}{\text{EJERCICIO 2.}}$ Sea una muestra aleatoria $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2) $.
# 
# $\color{purple}{\text{SOLUCIÓN}}$
# 
# Queremos encontrar el estimador de máxima verosimilitud para $\mu$ y $\sigma^2$.
# 
# La función de densidad es:
# $$ f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \cdot \mathbf{1}_{(-\infty, \infty)}(x)$$
# 
# Dado que $L(x_i;\mu,\sigma^2)= \prod_{i=1}^n f(x_i;\mu,\sigma^2)$
# 
# Se tiene que
# $$L(x_i;\mu,\sigma^2)= \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) $$
# 
# Aplicando el logaritmo
# 
# \begin{align*}
# ln[L(x_i;\mu,\sigma^2)] & = \sum_{i=1}^n ln\left[ \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)\right] \\
#                         & = -\frac{n}{2}ln(2\pi\sigma^2)-\frac{1}{2}\sum_{i=1}^n\frac{(x_i-\mu)^2}{\sigma^2}
# \end{align*}
# 
# Por un lado, aplicando la derivada parcial sobre $\mu$ tenemos
# 
# \begin{align*}
# \frac{\partial{(ln[L(x_i;\mu,\sigma^2)])}}{\partial \mu} & = \frac{\partial}{\partial \mu}\left[-\frac{n}{2}ln(2\pi\sigma^2)-\frac{1}{2}\sum_{i=1}^n\frac{(x_i-\mu)^2}{\sigma^2}\right] \\
#                                                          & = \sum_{i=1}^n \frac{(x_i-\mu)}{\sigma^2}
# \end{align*}
# 
# Igualando a $0$ se tiene
# 
# $$\sum_{i=1}^n \frac{(x_i-\mu)}{\sigma^2}=0$$
# 
# $$\therefore{\hat\mu} = \frac{1}{n}\sum_{i=1}^n x_i$$
# 
# $$\implies\hat\mu=\bar{x}$$
# 
# Por otro lado, aplicando la derivada parcial sobre $\sigma^2$ se tiene
# 
# \begin{align*}
# \frac{\partial{(ln[L(x_i;\mu,\sigma^2)])}}{\partial \sigma^2} & = \frac{\partial}{\partial \sigma^2}\left[-\frac{n}{2}ln(2\pi\sigma^2)-\frac{1}{2}\sum_{i=1}^n\frac{(x_i-\mu)^2}{\sigma^2}\right] \\
#                                                              & = -\frac{n}{2\sigma^2}+\frac{\sum_{i=1}^n (x_i-\mu)^2}{2\sigma^4}
# \end{align*}
# 
# Igualando a cero se tiene
# 
# $$-\frac{n}{2\sigma^2}+\frac{\sum_{i=1}^n (x_i-\mu)^2}{2\sigma^4}=0$$
# 
# $$\therefore{\hat\sigma^2} = \frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2$$
# 
# $$\implies{\hat\sigma^2}=\frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})^2$$

# $\color{purple}{\text{EJERCICIO 3.}}$ Suponga que la vida útil de los focos de cierta marca, se modela mediante una distribución exponencial de parámetro $\theta$ (desconocido). Probamos 5 focos y encontramos que tienen una vida útilde $2, 3, 1, 3$ y $5$ años, respectivamente. ¿Cuál es el MLE para $\theta$?

# $\text{Supongamos que la vida útil de los focos sigue una distribución exponencial con parámetro } \theta.$
# 
# $\text{Las observaciones son: } x_1 = 2, \; x_2 = 3, \; x_3 = 1, \; x_4 = 3, \; x_5 = 5.$
# 
# $\text{Determinar el MLE para Θ}$
# 
# $\color{purple}{\text{SOLUCIÓN}}$
# 
# $\text{La función de verosimilitud es:}$
# $$
# f(x_1, x_2, x_3, x_4, x_5 \mid \theta) = \theta^5 e^{- \theta \sum_{i=1}^5 x_i}.
# $$
# 
# $\text{Sustituyendo } \sum_{i=1}^5 x_i = 14 \text{, la función de verosimilitud se convierte en:}$
# $$
# f(2, 3, 1, 3, 5 \mid \theta) = \theta^5 e^{-14\theta}.
# $$
# 
# $\text{Tomamos el logaritmo natural de la función de verosimilitud para simplificar el cálculo:}$
# $$
# \ln(f(2, 3, 1, 3, 5 \mid \theta)) = 5 \ln(\theta) - 14\theta.
# $$
# 
# $\text{Calculamos la derivada de la log-verosimilitud respecto a } \theta:$
# $$
# \frac{d}{d\theta} \ln(f(2, 3, 1, 3, 5 \mid \theta)) = \frac{5}{\theta} - 14.
# $$
# 
# $\text{Igualamos la derivada a cero para maximizar la verosimilitud:}$
# $$
# \frac{5}{\theta} - 14 = 0.
# $$
# 
# $\text{Resolviendo para } \theta:$
# $$
# \theta = \frac{5}{14}.
# $$
# 
# $\text{Por lo tanto, el estimador de máxima verosimilitud (MLE) para } \theta \text{ es:}$
# $$
# \hat{\theta} = \frac{5}{14}.
# $$
# 

# In[13]:


import sympy as sp

# Definir la variable simbólica para theta
theta = sp.Symbol('theta', positive=True)

# Definir la función log-likelihood
log_likelihood = 5 * sp.log(theta) - 14 * theta

# Derivar la log-likelihood respecto a theta
log_likelihood_derivative = sp.diff(log_likelihood, theta)

# Resolver la ecuación de la derivada igual a 0
theta_mle = sp.solve(log_likelihood_derivative, theta)

# Resultado
print(f"El MLE para theta es: {theta_mle[0]}")


# Este ejercicio también se puede comprobar con el siguiente codigo, que calcula el reciproco de la media:
# 

# In[14]:


import numpy as np

# Datos de vida útil de los focos (en años)
vida_util = [2, 3, 1, 3, 5]

# Calcular la media de los datos
media = np.mean(vida_util)

# Calcular el MLE de theta
theta_mle = 1 / media

print(f"El MLE para theta es: {theta_mle}")


# $\color{purple}{\text{EJERCICIO 4.}}$ Sea $ (X_1, \ldots, X_n) $ una muestra aleatoria con función de densidad:
# $f(x, \theta) = \theta(1+x)^{-(1+\theta)} \quad \text{para} \quad x > 0, \theta > 0$
# 
# $\color{purple}{\text{SOLUCIÓN.}}$
# 
# Estimar $\theta$ por momentos asumiendo que  $ \theta > 1$:
# 
# $$E[X] = \left. \frac{dM_X(t)}{dt} \right|_{t=0}$$
# 
# \begin{align*} 
#  \frac{dM_X(t)}{dt} & = \frac{d}{dt}\int_{1}^{\infty} e^{tx}  \theta(1+x)^{-(1+\theta)} \, dx \\
#                     & = \int_{1}^{\infty} xe^{tx}  \theta(1+x)^{-(1+\theta)} \, dx
# \end{align*}
# 
# $$\frac{dM_X(0)}{dt} =\frac{d}{dt}\int_{1}^{\infty} x \theta(1+x)^{-(1+\theta)} \, dx$$
# 
# donde
# 
# $$ u = 1 + x \\ du = dx \\ $$
# 
# \begin{align*}
# \implies \frac{dM_X(0)}{dt} & = \theta \int_{1}^{\infty} (u - 1)u^{-(1+\theta)} \, du \\
#                             & = \theta  \int_{1}^{\infty} u^{-\theta} \ -  u^{-(1+\theta)} \, du \\
#                             & = \theta \left(\frac{1}{\theta -1} - \frac{1}{\theta}\right)
# \end{align*}
# 
# $E[X] = \frac{1}{\theta - 1} \quad \text{( } \theta > 1\text{)}$
# 
# Por momentos 
# 
# $E[X] = \overline{X} =\frac{1}{\theta - 1} \quad  \implies \theta = 1 + \frac{1}{\overline{x}} $

# ## Distribuciones muestrales

# ### 1. Distribución *t* de Student
# 
# Sean:
# 
# - $X \sim \mathcal{N}(0,1)$  
# - $Y \sim \chi^2(n - 1)$  
# 
# Entonces:
# 
# $$
# T = \frac{X \cdot \sqrt{n}}{\sqrt{\dfrac{Y}{n - 1}}} \sim t(n - 1)
# $$
# 
# Gráficamente, la distribución *t* tiene forma de campana, similar a la normal estándar, pero con colas más gruesas. A medida que $n \to \infty$, se aproxima a una normal estándar:
# 
# $$
# \lim_{n \to \infty} t(n) = \mathcal{N}(0,1)
# $$
# 
# 
# >Esta distribución es útil cuando se desconoce la varianza poblacional y se utiliza una estimación muestral.
# 

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# Valores de x para graficar
x = np.linspace(-5, 5, 500)

# Distribuciones t con diferentes grados de libertad
dfs = [1, 3, 10, 30]
colors = ['red', 'orange', 'green', 'blue']

plt.figure(figsize=(10, 6))

# Graficar curvas t
for df, color in zip(dfs, colors):
    plt.plot(x, t.pdf(x, df), label=f't(df={df})', color=color)

# Graficar la normal estándar
plt.plot(x, norm.pdf(x), 'k--', label='Normal estándar')

# Formato de la gráfica
plt.title('Comparación entre la distribución t y la normal estándar', fontsize=14)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Parámetros del ejemplo
n = 16                     # tamaño de muestra
df = n - 1                 # grados de libertad
media_muestral = 17.5
media_hipotetica = 16
desv_est_muestral = 3

# Estadística t
t_stat = (media_muestral - media_hipotetica) / (desv_est_muestral / np.sqrt(n))
print(f'Estadística t: {t_stat:.3f}')

# Intervalo de confianza del 95%
alpha = 0.05
t_critico = t.ppf(1 - alpha/2, df)
print(f'Valor crítico t (95%): ±{t_critico:.3f}')

# Gráfica
x = np.linspace(-5, 5, 500)
y = t.pdf(x, df)

plt.figure(figsize=(10,6))
plt.plot(x, y, label=f't(df={df})', color='blue')

# Rellenar área bajo la curva para el IC
plt.fill_between(x, y, where=(x >= -t_critico) & (x <= t_critico), color='lightblue', alpha=0.6, label='Área 95%')

# Línea para el valor de la estadística t calculada
plt.axvline(t_stat, color='red', linestyle='--', lw=2, label=f'Estadístico t = {t_stat:.2f}')
plt.axvline(-t_critico, color='gray', linestyle='--')
plt.axvline(t_critico, color='gray', linestyle='--')

# Detalles del gráfico
plt.title('Distribución t de Student con intervalo de confianza del 95%', fontsize=14)
plt.xlabel('$t$', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# ### 2. Distribución F de Fisher
# 
# Sea:
# 
# $$
# F = \frac{\chi^2_{(m-1)} / (m - 1)}{\chi^2_{(n-1)} / (n - 1)} \sim F(m - 1,\ n - 1)
# $$

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, chi2

# Rango de valores para las gráficas
x_f = np.linspace(0, 5, 500)
x_chi = np.linspace(0, 20, 500)

# --- Distribución F de Fisher ---
df1, df2 = 5, 10
f_dist = f.pdf(x_f, df1, df2)

plt.figure(figsize=(12, 5))

# Gráfica de F
plt.subplot(1, 2, 1)
plt.plot(x_f, f_dist, label=f'F({df1}, {df2})', color='purple')
plt.title('Distribución F de Fisher', fontsize=14)
plt.xlabel('x')
plt.ylabel('Densidad')
plt.grid(True)
plt.legend()

# --- Distribuciones Ji-cuadrada ---
df_chi_list = [1, 3, 6, 10]

plt.subplot(1, 2, 2)
for df in df_chi_list:
    plt.plot(x_chi, chi2.pdf(x_chi, df), label=f'χ²({df})')
plt.title('Distribuciones Ji-cuadrada', fontsize=14)
plt.xlabel('x')
plt.ylabel('Densidad')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ### 3️. Distribución Ji-cuadrada (Chi-cuadrada)
# 
# Si:
# 
# - $X \sim \mathcal{N}(0,1)$  
# - Entonces: $X^2 \sim \chi^2(1)$
# 
# Y si:
# 
# $$
# \sum_{i=1}^{n} X_i^2 \sim \chi^2(n)
# $$
# 
# > La distribución $\chi^2(k)$ surge como la suma de los cuadrados de $k$ variables normales estándar independientes.
# 

# ## 3.2 Intervalos de Confianza

# In[6]:


# Librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# ### Intervalo de Confianza para la media de una distribución normal (σ conocida)
# 
# Sea $X_1, \dots, X_n$ una m.a. $X_i \sim \mathcal{N}(\mu, \sigma^2)$. Entonces:
# 
# - $\bar{X} \sim \mathcal{N}(\mu, \frac{\sigma^2}{n})$
# - $Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1)$
# 
# ### Nivel de significancia (error): $\alpha$
# 
# El intervalo de confianza se basa en que:
# $$
# \mathbb{P}\left( -z_{\alpha/2} < \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} < z_{\alpha/2} \right)
# = \mathbb{P}\left( \bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}} < \mu < \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \right)
# = 1 - \alpha
# $$
# 
# ### El intervalo del $(1 - \alpha) \cdot 100\%$ de confianza para $\mu$ (con $\sigma$ conocida) es:
# 
# $$
# \left( \bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \quad \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \right)
# $$
# donde
# 
# - $\sigma$ = desviación estándar  
# - $\frac{\sigma}{\sqrt{n}}$ = error estándar de la media  
# - $z_{\alpha/2}$ determina el nivel de confianza  
# - El intervalo está centrado en $\bar{X}$
# 
# $\color{slateblue}{\text{OBSERVACIÓN.}}$
# 
# - A mayor $n$, menor es el error estándar, y el intervalo de confianza es más pequeño.
# - Esto hace que se acerque más a la media $\mu$, es decir, que haya menor variación.
# 
# 
# $\color{indigo}{\text{EJEMPLO 1.}}$ Para tratar de estimar la media de consumo por cliente en un gran restaurante, se reunieron datos de una muestra de 49 clientes durante 3 semanas.
# 
# **a)** Supongamos que la **desviación estándar de la población** es de $2.50. ¿Cuál es el error estándar de la media?
# 
# **b)** Con un nivel de confianza del 95%. ¿Cuál es el margen de error?
# 
# **c)** Si la **media de la muestra** es de $22.60, ¿cuál es el intervalo de confianza del 95% para la media de la población?

# In[7]:


# Datos
n=49
sigma = 2.50
media_muestral = 22.60
confianza = 0.95

#a) Error estándar de la media
error_estandar = sigma/np.sqrt(n)
print(f"a) Error estándar de la media: {error_estandar:.2f}")

# b) Margen de error
z = st.norm.ppf(1-(1-confianza)/2)
margen_error = z * error_estandar
print(f"b) Margen de error con 95% de confianza: {margen_error:.2f}")

# c)
limite_inferior = media_muestral - margen_error
limite_superior = media_muestral + margen_error
print(f"c) Intervalo de confianza del 95%: ({limite_inferior:.2f},{limite_superior:.2f})")


# $1-\alpha = .95$, implica que $z_{\alpha/2} = z_{0.025} = 1.96$
# 
# $\color{indigo}{\text{EJEMPLO 2.}}$ Supongamos que se toma una muestra aleatoria de 100 personas para estimar la media del peso de una población y se obtiene que la media muestral es de 70kg con una desviación estándar que es conocida de 10kg. Para un nivel de confianza del 95%, calcular el intervalo de confianza.

# In[8]:


# Datos

n = 100
media_muestral = 70
sigma = 10
nivel_confianza = 0.95

error_estandar = sigma / np.sqrt(n)

# Utilizaremos la función scipy.stats.norm.interval
# Sintaxis
# scipy.stats.norm.interval(confidence, loc = media, scale = error_estandar)

intervalo = st.norm.interval(confidence=nivel_confianza, loc = media_muestral, scale = error_estandar)
print(f"Intervalo de confianza del 95% es:{intervalo}")


# In[9]:


# Puntos para graficar la curva normal
x = np.linspace(media_muestral - 4*error_estandar, media_muestral + 4*error_estandar, 500)
y = st.norm.pdf(x, loc=media_muestral, scale=error_estandar)

# Gráfica
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Distribución normal', color='black')

# Sombrear el intervalo de confianza
plt.fill_between(x, y, where=(x >= intervalo[0]) & (x <= intervalo[1]), color='skyblue', alpha=0.6, label='IC 95%')

# Líneas verticales
plt.axvline(intervalo[0], color='blue', linestyle='--', label=f'IC inferior = {intervalo[0]:.2f}')
plt.axvline(intervalo[1], color='blue', linestyle='--', label=f'IC superior = {intervalo[1]:.2f}')
plt.axvline(media_muestral, color='red', linestyle='-', label=f'Media muestral = {media_muestral}')

# Estética
plt.title('Intervalo de confianza del 95% para la media ($\\sigma$ conocida)', fontsize=14)
plt.xlabel('Valor de la variable')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.show()


# ### Intervalo de Confianza para la media $\mu$ de una distribución normal ($\sigma$ deconocida)
# 
# Sea $X_1, \dots, X_n$ una m.a. $X_i \sim \mathcal{N}(\mu, \sigma^2)$. Entonces:
# 
# - $\bar{X} \sim \mathcal{N}(\mu, \frac{\sigma^2}{n})$
# - $Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1)$
# 
# Si el tamaño de la muestra es menor a 30, se utiliza la $t$:
# 
# ### Estadístico t
# 
# Definimos el estadístico $t$ de la siguiente manera:
# $$T = \frac{\bar{X}-\mu}{s/\sqrt{n}} \sim t_{(n-1)} $$
# 
# ### El intervalo del $(1 - \alpha) \cdot 100\%$ de confianza para $\mu$ (con $\sigma$ desconocida) es:
# 
# $$
# \left( \bar{X} - t_{\alpha/2} \frac{s}{\sqrt{n}}, \quad \bar{X} + t_{\alpha/2} \frac{s}{\sqrt{n}} \right)
# $$
# donde
# 
# - $s$ = desviación estándar muestral
# - $\frac{s}{\sqrt{n}}$ = error estándar de la media  
# - $t_{\alpha/2}$ determina el nivel de confianza  
# - El intervalo está centrado en $\bar{X}$

# $\color{indigo}{\text{EJEMPLO 3.}}$ Supongamos que tenemos los siguientes datos
# $$ datos = [45, 55, 67, 45, 68, 79, 98, 87, 84, 82] $$
# Calcular un intervalo de confianza para la media.

# In[10]:


# Datos del ejemplo 
data = [45, 55, 67, 45, 68, 79, 98, 87, 84, 82]
confidence = 0.95
gl = len(data) - 1 # grados de liber

# Media y error estandar
mean = np.mean(data)
error_est = st.sem(data)

# Intervalo de confianza usando t de Student
intervalo = st.t.interval(confidence, gl, loc = mean, scale = error_est)

print(f"Media muestral: {mean:.2f}")
print(f"Error estandar: {error_est:.2f}")
print(f"El intervalo de confianza del 95% es: {intervalo}")


# $\color{indigo}{\text{EJEMPLO 4.}}$ Los artículos de cerámica utilizados sobre velas electricas sobrecargadas se rompen con diferentes presiones. Supongamos que los datos provienen de una distribución normal.
# 
# La resistencia a la ruptura fue medida en una muestra de 100 artículos, y el promedio fue de $1750$ con un desviación estándar de 315.8
# 
# **a)** Estimar con un nivel del confianza del $90\%$ a la media poblacional de la presión de la ruptura.
# 
# **b)** Estimar con un nivel del confianza del $90\%$ a la varianza poblacional.

# In[11]:


from scipy.stats import chi2
# datos
n = 100
media_muestral = 1750
desv_estandar_muestral = 315.8
confianza = 0.90

# a) Usar la normal ya que el tamaño de la muestra es grande
error_est = desv_estandar_muestral / np.sqrt(n)
error_est
intervalo_media = st.norm.interval(confidence=confianza, loc = media_muestral, scale = error_est)
print(f"El intervalo de confianza del 90% es: {intervalo_media}")

# b) Intervalo de confianza para la varianza (usar chi-cuadrada)
alpha = 1-confianza
gl = n-1
s2 = desv_estandar_muestral**2

#Cuantiles de la chi-cuadrada
chi2_inf = st.chi2.ppf(alpha / 2,df=gl)
chi2_sup = st.chi2.ppf(1- alpha / 2,df=gl)

# Intervalo de confianza
intervalo_varianza = ((gl * s2) / chi2_sup, (gl * s2) / chi2_inf )
print(f"El intervalo de confianza del 90% es: {intervalo_varianza}")

#Otra forma
chi2_low, chi2_high = chi2.interval(confianza,df=gl)
intervalo_varianza1 = ((gl * s2) / chi2_high, (gl * s2) / chi2_low )
print(f"El intervalo de confianza del 90% es: {intervalo_varianza1}")


# $\color{indigo}{\text{EJEMPLO 5.}}$ El artículo *"Evaluation of a Ventilation Strategy to Prevent Barotrauma in Patients at High Risk for Acute Respiratory Distress Syndrome"* reportó sobre un experimento con 120 pacientes con anestesistas en cuidados intensivos (UCI), los cuales fueron divididos al azar en dos grupos, donde cada uno esta compuesto por 60 pacientes. 
# 
# - Grupo A: promedio de permanencia = 14.1 horas
# - Grupo B: promedio de permanencia = 17.5 horas
# - Desviación estándar en ambos = 5.1 hrs
# 
# Encontrar un intervalo del $95\%$ de confianza para la diferecia de medias poblacionales: $(\mu_A - \mu_B)$  

# In[12]:


#Datos 
n1 = 60
n2 = 60
media1 = 14.1
media2 = 17.5
sigma = 5.1
confianza = 0.95
alpha = 1-confianza
#gl = n1 + n2 -2

#valor critico
z = st.norm.ppf(1 - alpha / 2)

#Error estandar
error_est = sigma * np.sqrt(1/n1 + 1/n2)
dif_medias = media1 - media2
margen_error = z *error_est

lim_inf = dif_medias - margen_error
lim_sup = dif_medias + margen_error
print(f"Diferencia de las medias: {dif_medias:.2f}")
print(f"Intervalo de confianza del 95%: ({lim_inf:.2f},{lim_sup:.2f})")


# ## Propiedad de Invarianza
# 
# Si $\hat{\theta}$ es EMV de ${\theta}$ pero nos interesa estimar una funcion de ${\theta}$, digamos T(${\theta}$)
# 
# $\hat{T(\theta)} = T(\hat \theta)$
# 
# si $\hat{\theta} = \bar{x}$ es EMV de ${\theta}$
# 
# $\implies {T(\theta)} = \theta ^ 2 \implies \hat{T(\theta)} = (\bar{x})^ 2$
# 
# $ X_1, \dots, X_n \sim \text{exp}(\lambda)$
# 
# ### Estimadores:
# 
# $\hat \theta_1 = x_1$
# 
# $\hat \theta_2 = \frac {x_1 + x_2} {2}$
# 
# $\hat \theta_3 = \frac {x_1 + 2x_2} {3}$
# 
# ### Error cuadrático medio
#  $ECM (\hat \theta) = \mathbb{E}[(\hat \theta - \theta)^ 2]$
# 
#  $=Var(\hat \theta) + \mathbb{E}[(\hat \theta - \theta)]^ 2$  
# 
# 
# 
# $\color{indigo}{\text{EJEMPLO.}}$ Sea $ X_1, \dots, X_n \sim \text{Poisson}(\lambda)$
# 
# a) Encontrar EMV 
# 
# b) ¿Es insesgado?
# 
# c) ECM
# 
# ---
# 
# a) $L(x_1, \dots, x_n, \theta) = \prod_{i=1}^{n} \frac{e^{-\theta} \theta^{x_i}}{x_i!}
# = \frac{e^{-n\theta} \theta^{\sum x_i}}{x_1! x_2! \dots x_n!}$
# 
# $ \implies \ln L(x_1, \dots, x_n; \theta) 
# = \ln \left( \frac{e^{-n\theta} \theta^{\sum x_i}}{x_1! \dots x_n!} \right)
# = \ln(e^{-n\theta} \text{ } \theta^{\sum x_i}) - \ln(x_1! \dots x_n!)$
# 
# $= -n\theta + \sum_{}^n x_i \ln(\theta) - \sum_{}^n \ln(x_i!)$
# 
# $ \implies \frac{\partial \ln L(x_1, \dots, x_n; \theta)}{\partial \theta} = -n + \sum x_i \cdot \frac{1}{\theta} = 0 $
# 
# $\implies \frac{1}{\theta} \sum x_i = n 
# \implies \frac{\sum x_i}{n} = \theta \\
# \implies \hat{\theta} = \bar{x}$
# 
# ---
# 
# b) $\mathbb{E}[\hat{\theta}] = \mathbb{E}\left[\frac{\sum x_i}{n}\right] 
# = \frac{1}{n} \mathbb{E}[\sum x_i] 
# = \frac{\sum \mathbb{E}[x_i]}{n} 
# = \frac{\sum_{}^n \theta}{n} 
# = \frac{n \theta}{n} = \theta$
# 
# ---
# 
# c) $\text{ECM}(\hat{\theta}) = \text{Var}(\hat{\theta}) 
# = \text{Var}\left( \frac{\sum x_i}{n} \right) 
# = \frac{1}{n^2} \text{Var}\left( \sum x_i \right)
# = \frac{\sum \text{Var}(x_i)}{n^2}
# = \frac{\sum \theta}{n^2}
# = \frac{n\theta}{n^2} = \frac{\theta}{n}$
# 
# ${\text{ECM}(\hat{\theta}) = \frac{\theta}{n}}$
# 
# 
# ### Def. El sesgo  B  de un estimador $ \hat{\theta} $
# 
# $B = \mathbb{E}[\hat{\theta}] - \theta $
# 
# 1) Si $\mathbb{E}[\hat{\theta}] > \theta $, el estimador es *sesgado positivamente*  (a)
# 
# 2) Si $\mathbb{E}[\hat{\theta}] < \theta $, el estimador es *sesgado negativamente*  (b)
# 
# ### Minimización del ECM
# 
# Sea$X $ una variable aleatoria con segundo momento de orden finito.
# 
# Definimos la función:
# $ g(u) = \mathbb{E}[(X - u)^2] \quad \Leftarrow \text{ECM} $
# 
# Entonces:
# $ g(u) \text{ se minimiza cuando } \mathbb{E}[X] = u $
# 
# ### Consecuencia:
# 
# $ \text{Var}(X) \leq \mathbb{E}[(X - u)^2]$
# 
# ---
# ### Estadísticas Suficientes
# 
# Sea $ X_1, \ldots, X_n $ una m.a. con densidad $ f(\mathrel{\cdot}; \theta) $, $ \theta $ puede ser un vector.
# 
# Una estadística $ T = t(X_1, \ldots, X_n) $ es una *estadística suficiente* ⟺  
# la distribución de $ X_1, \ldots, X_n $ dado $ T = t $ no depende de $ \theta $.
# 
# $
# \mathbb{P}(X_1 = t_1, X_2 = t_2, \ldots, X_n = t_n \mid T = t) \text{ no depende de } \theta
# $
# 
# 
# ### Teorema de Fisher-Neyman
# 
# Si $ X_1, \ldots, X_n $ es una m.a. con f.d. $ f(x; \theta) $  
# Sea $ T = t(X_1, \ldots, X_n) $ una estadística cuya f.d.p. sea $ g(t; \theta) $  
# 
# Entonces $ T $ es suficiente para $ \theta $ si y sólo si:
# 
# $
# L(X_1, \ldots, X_n; \theta) = H(X_1, \ldots, X_n) \cdot g(t(X_1, \ldots, X_n), \theta)
# $
# 
# donde:
# 
# - $ H(X_1, \ldots, X_n) $ *no depende de* $ \theta $
# 
# 
# ### Estimadores → UMVUE(Estimadores insesgados de varianza uniformemente mínima)
# 
# Sea $ X_1, \ldots, X_n $ una m.a. de $ f(\mathrel{\cdot}; \theta) $ un estimador $ T^* = t(X_1, \ldots, X_n) $  
# de $ T(\theta) $ se define como un *UMVUE* de $ \tau(\theta) $ si y sólo si:
# 
# 1. $ \mathbb{E}[T^*] = T(\theta) $ (insesgado)  
# 2. $ \text{Var}(T^*) \leq \text{Var}(T) $ para cualquier otro estimador insesgado $ T = t(X_1, \ldots, X_n) $ de $ T(\theta) $
# 
# ---
# 
# ### Desigualdad de Cramér–Rao
# 
# Establece un límite inferior para la varianza de cualquier estimador
# 
# $\text{Var}(T) \geq \frac{[T'(\theta)]^2}{n \mathbb{E} \left[ \left( \frac{\partial \ln f(x_i; \theta)}{\partial \theta} \right)^2 \right]}$
# 
# Si $ \text{Var}(T) $ alcanza el límite inferior:  
# $\Rightarrow \text{estimador insesgado óptimo}$
# 
# ## Intervalos de Confianza
# 
# $\hat{\theta} = T(X_1, \ldots, X_n) \quad \longleftarrow \text{estimación puntual}$
# 
# $\mathbb{P}(a < \theta < b) = \delta \in (0,1)$
# 
# $a, b \rightarrow \text{cantidades pivotales}$
# 
# ### Definición
# 
# Sea $ X_1, \ldots, X_n $ una m.a. con f.d. $f(x; \theta) $. Sean  
# $ T_1 = t_1(X_1, \ldots, X_n) $ y $ T_2 = t_2(X_1, \ldots, X_n) $ dos estadísticos tales que $ T_1 \leq T_2 $
# 
# $\mathbb{P}\left( T_1 \leq \tau(\theta) \leq T_2 \right) = \delta$
# 
# Al intervalo aleatorio $ (T_1, T_2) $ se le llama *intervalo del $ \delta \times 100\% $* de confianza.
# 
# ### Intervalos de Confianza (I.C.)
# 
# ### Para la media de una distribución normal
# - *σ conocida*
#   $\text{Normal} $
# 
# - *σ desconocida*
#   $ t $
# 
# ### Para la varianza $ \sigma^2 $ de una distribución normal
# - *μ conocida*
#   $ \chi^2 $
# 
# - *μ desconocida*
#   $ F $
# 
# ### Para la diferencia de medias $ \mu_A - \mu_B $ de dos poblaciones normales
# - *σ conocida*
# - *σ desconocida*
# 
# ### Para el cociente de varianzas $ \sigma_A^2 / \sigma_B^2 $
# - *μ conocida o desconocida*

# ### 3.3 Pruebas de Hipótesis
# 
# ### Intervalos de Confianza
# 
# Sea $X_1, \ldots, X_n $ una muestra aleatoria (m.a.) donde $ X_i \sim \mathcal{N}(\mu, \sigma^2) $.  
# 
# ### Error Estándar (SE)
# El error estándar mide cuánto varía una estadística.  
# 
# - *Si la desviación estándar poblacional $ \sigma $ es conocida y el tamaño de la muestra es $ n $:*  
#   $
#   SE = \frac{\sigma}{\sqrt{n}}
#   $  
# 
# - *Si $ \sigma $ es desconocida, se usa la desviación estándar muestral $ S $:*  
#   $  SE = \frac{S}{\sqrt{n}} $  
# 
# - *Propiedad:* A mayor tamaño de muestra $ n $, menor es el error estándar.  
# 
# ### Intervalo de Confianza para la Media $ \mu $ de una Distribución Normal
# $ \hat M = \bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right), \quad \text{con} \quad \text{Var}(X_i) = \sigma^2 $
# 
# $ Z = \frac{\bar{X} - \mu}{\sqrt{\sigma^2/n}} = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1) $
# 
# Nivel de Confianza (1-α)
# 
# $ \mathrm{IC}α = \left[ \bar{X} - Z{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}, \quad \bar{X} + Z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \right] $
# 
# - $ Z_{\alpha/2} $ es el valor crítico de la distribución normal estándar que deja $ \alpha/2 $ en cada cola.
# 
# 
# ### Intervalo de Confianza para la Varianza $\sigma^2 $ de una Distribución Normal (µ desconocida)
# 
# $\frac{(n-1)S^2}{\chi^2_{\frac{\sigma}{2},n-1}} < \sigma^2 < \frac{(n-1)S^2}{\chi^2_{1-\alpha/2,n-1}}$ 
# 
# ### Intervalo de Confianza para la Diferencia de Medias $ \mu_A - \mu_B $
# 
# Supuestos
# - Muestras independientes:
#   - $ X_1, \ldots, X_n $ m.a. de $\mathcal{N}(\mu_A, \sigma^2) \\ \overline{X}_A \sim \mathcal{N}\left(\mu_A, \frac{\sigma^2}{n}\right)$
# 
#   - $ Y_1, \ldots, Y_m $ m.a. de $ \mathcal{N}(\mu_B, \sigma^2) \\ \quad \overline{X}_B \sim \mathcal{N}\left(\mu_B, \frac{\sigma^2}{m}\right)$
# 
# ### Varianza poblacional $ \sigma^2 $ *conocida* 
# 
# ${X}_A - {X}_B \sim \mathcal{N}\left(\mu_A - \mu_B, \sigma^2 \left( \frac{1}{n} + \frac{1}{m} \right) \right)$
# 
# Intervalo del (1-α)×100% de Confianza para $\mu_A - \mu_B$ es:
# $\left( \overline{X}A - \overline{X}_B \right) \pm Z{1-\alpha/2} \cdot \sigma \sqrt{\frac{1}{n} + \frac{1}{m}}$
# 
# ### Varianza poblacional $ \sigma^2 $ *desconocida* 
# 
# $T = \frac{(\overline{X}_A - \overline{X}_B) - (\mu_A - \mu_B)}{\sqrt{\frac{n + m}{nm}} \cdot \sqrt{\frac{\sum (X_i - \overline{X}_A)^2 + \sum (Y_i - \overline{X}_B)^2}{n + m - 2}}} \sim t(n + m - 2)$
# 
# $ IC = (\overline{X}A - \overline{X}_B) \pm t{1 - \alpha/2}^{(n + m - 2)} \cdot \sqrt{\frac{\sum (X_i - \overline{X}_A)^2 + \sum (Y_i - \overline{X}_B)^2}{n + m - 2}}$
# 
# $IC \implies$ proporción $ p $ de una Binomial
# 
# $I = {P_o} \pm z_{\alpha/2} \cdot \sqrt{\frac{{P_0}(1 - {P_0})}{n}}$
# 
# ### Pruebas de Hipótesis
# 
# - *Hipótesis Nula (H₀):* Determina la distribución.
# - *Hipótesis Alternativa (H₁ o Hₐ):* La que no es nula.
# 
# $Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1)$
# 
#    - H₀: $ \mu \leq \mu_0 $ vs H₁: $ \mu > \mu_0 $  
#    - H₀: $ \mu \geq \mu_0 $ vs H₁: $ \mu < \mu_0 $  
#    - H₀: $ \mu = \mu_0 $ vs H₁: $ \mu \neq \mu_0 $

# ### Ejercicios Intervalos de Confianza

# In[18]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# $\color{indigo}{\text{EJEMPLO 1.}}$ En una encuesta de una Universidad, 225 estudiantes son seleccionados al azar de los que toman cálculo y se les pregunta si cálculo es su materia favorita. 100 estudiantes responden que el cálculo es su materia favorita. Proporcione un intervalo de confianza del 95 %  para la proporción de todos los estudiantes que toman cálculo y que la consideran su materia favorita.

# In[19]:


n1 = 225
exitos1 = 100
p_hat = exitos1 / n1
confianza1 = 0.95
z_valor1 = st.norm.ppf(1 - (1 - confianza1)/2)

error_estandar1 = np.sqrt(p_hat * (1 - p_hat) / n1)
margen_error1 = z_valor1 * error_estandar1

n1 = 225
exitos1 = 100
p_hat = exitos1 / n1
confianza1 = 0.95
z_valor1 = st.norm.ppf(1 - (1 - confianza1)/2)

error_estandar1 = np.sqrt(p_hat * (1 - p_hat) / n1)
margen_error1 = z_valor1 * error_estandar1

ic1a=(p_hat - margen_error1,p_hat + margen_error1)

print(f"Intervalo al 95%: [{ic1a[0]:.4f}, {ic1a[1]:.4f}]")


# $\color{indigo}{\text{EJEMPLO 2.}}$
# 
# Los datos $1.2,\ 2.1,\ 2.3,\ 1.9,\ 3.0,\ 1.5,\ 2.8,\ 2.4,\ 1.7,\ 2.6,\ 2.0,\ 1.8,\ 3.1,\ 2.2,\ 2.5$ se toman a partir de una distribución normal $N(\mu,\sigma^2)$ con $\mu$ desconocida.
# 
# **a)** Encuentra un intervalo de confianza al nivel de $90\%$ para $\mu$, dado que $\sigma =2$.
# 
# **b)** Encuentra un intervalo de confianza al nivel de $90\%$ para $\mu$.
# 
# **c)** Encuentra un intervalo de confianza al nivel de $90\%$ para $\sigma^2$.

# In[20]:


datos2 = np.array([1.2, 2.1, 2.3, 1.9, 3.0, 1.5, 2.8, 2.4, 1.7, 2.6, 2.0, 1.8, 3.1, 2.2, 2.5])
n2 = len(datos2)
media2 = np.mean(datos2)
s2 = np.std(datos2, ddof=1)
confianza2 = 0.90

# a)
sigma_a = 2
z_valor2a = st.norm.ppf(1 - (1 - confianza2)/2)
error_estandar2a = sigma_a / np.sqrt(n2)
margen_error2a = z_valor2a * error_estandar2a
ic2a = (media2 - margen_error2a, media2 + margen_error2a)

print("\na)")
print(f"   Intervalo: [{ic2a[0]:.4f}, {ic2a[1]:.4f}]")

# b)
t_valor2b = st.t.ppf(1 - (1 - confianza2)/2, df=n2-1)
error_estandar2b = s2 / np.sqrt(n2)
margen_error2b = t_valor2b * error_estandar2b
ic2b = (media2 - margen_error2b, media2 + margen_error2b)

print("\nb)")
print(f"   Intervalo: [{ic2b[0]:.4f}, {ic2b[1]:.4f}]")


# Parte c: Intervalo para varianza
chi2_inf2c = st.chi2.ppf((1 - confianza2)/2, df=n2-1)
chi2_sup2c = st.chi2.ppf(1 - (1 - confianza2)/2, df=n2-1)
ic2c = ((n2-1)*s2**2/chi2_sup2c, (n2-1)*s2**2/chi2_inf2c)

print("\nc)")
print(f"   Intervalo: [{ic2c[0]:.4f}, {ic2c[1]:.4f}]")


# $\color{indigo}{\text{EJEMPLO 3.}}$
# 
# Los ingresos semanales promedio de las personas que trabajan en varias industrias aparecieron en el *The New York Times 1988 Almanac*.  
# Esos ingresos para quienes trabajan en los servicios fueron de $\$369$. Suponga que este resultado se basó en una muestra de 250 personas dedicadas a los servicios y que la desviación estándar de la muestra fue de $\$50$. Calcula el intervalo de confianza del 95% para la media de la población de ingresos semanales de personas que trabajan en los servicios.

# In[21]:


# Datos
n3 = 250
media_muestral3 = 369
sigma3 = 50
nivel_confianza3 = 0.95
error_estandar3 = sigma3 / np.sqrt(n3)


intervalo3 = st.norm.interval(confidence = nivel_confianza3, loc = media_muestral3, scale = error_estandar3)
intervalo3= (float(intervalo3[0]), float(intervalo3[1]))
print(f"Intervalo de confianza del 95% es: {intervalo3}")


# $\color{indigo}{\text{EJEMPLO 4.}}$
# 
# En un estudio de préstamos a estudiantes, el Departamento de Educación informó que los beneficiarios del fondo Stafford Loan deberían un promedio de \$12,658 al recibirse (*USA Today*, 5 de abril de 1995). Suponga que este promedio de deuda se basa en una muestra de 480 préstamos a estudiantes y que la desviación estándar de la población de las deudas al recibirse es \$2,000.
# 
# **a)** Determina un estimado de confianza del **90%** del promedio poblacional de la deuda.
# 
# **b)** Determina un estimado de confianza del **95%** del promedio poblacional de la deuda.
# 
# **c)** Determina un estimado de confianza del **99%** del promedio poblacional de la deuda.
# 
# **d)** Describe lo que sucede con el **ancho del intervalo de confianza** a medida que se **aumenta el nivel de confianza**. ¿Parece razonable? Explica tu respuesta.

# In[22]:


# Datos
n4 = 480
media_muestral4 = 12658
sigma4 = 2000
nivel_confianza4_95 = 0.95
nivel_confianza4_90 = 0.90
nivel_confianza4_99 = 0.99
error_estandar4 = sigma4 / np.sqrt(n4)

# a) 90%
intervalo4_90 = st.norm.interval(confidence = nivel_confianza4_90, loc = media_muestral4, scale = error_estandar4)
intervalo4_90= (float(intervalo4_90[0]), float(intervalo4_90[1]))
print("\nb) 90%:")
print(f"Intervalo de confianza del 90% es: {intervalo4_90}")

# b) 95%
intervalo4_95 = st.norm.interval(confidence = nivel_confianza4_95, loc = media_muestral4, scale = error_estandar4)
intervalo4_95= (float(intervalo4_95[0]), float(intervalo4_95[1]))
print("\nb) 95%:")
print(f"Intervalo de confianza del 95% es: {intervalo4_95}")

# c) 99%
intervalo4_99 = st.norm.interval(confidence = nivel_confianza4_99, loc = media_muestral4, scale = error_estandar4)
intervalo4_99= (float(intervalo4_99[0]), float(intervalo4_99[1]))
print("\nb) 99%:")
print(f"Intervalo de confianza del 99% es: {intervalo4_99}")


print("\nd) A medida que aumenta el nivel de confianza, el intervalo se hace más amplio.")
print("   Esto es razonable porque para tener mayor certeza de capturar el parámetro")
print("   poblacional, necesitamos un rango más amplio de valores posibles.")


# $\color{indigo}{\text{EJEMPLO 5.}}$
# 
# La encuesta anual de calidad de automóviles, efectuada por *J. D. Power & Associates*, determinó que la cantidad promedio de defectos, en todas las marcas, por cada vehículo nuevo es **1.07**  (*The Wall Street Journal*, 27 de enero de 1994). Suponga que se toma una muestra de **30 automóviles nuevos** de determinada marca y se obtienen las siguientes cantidades de defectos por vehículo:
# $$ 0, 1, 1, 2, 1, 0, 2, 3, 2, 1, 0, 2, 0, 0, 2, 3, 0, 4, 3, 1, 1, 1, 0, 2, 0, 2, 0, 3, 1, 0 $$
# 
# **a)** ¿Cuál es el promedio muestral de la cantidad de defectos por vehículo?
# 
# **b)** ¿Cuál es la desviación estándar de la muestra?
# 
# **c)** Determine un intervalo de confianza del 95% para la **media de defectos** por vehículo de esta marca.
# 
# **d)** Un analista sugirió que se debería **revisar una muestra mayor** antes de comparar con el promedio general de J.D. Power (1.07). ¿Respalda usted esta idea? ¿Por qué?

# In[23]:


defectos = np.array([0, 1, 1, 2, 1, 0, 2, 3, 2, 1, 0, 2, 0, 0, 2, 3, 0, 4, 3, 1, 1, 1, 0, 2, 0, 2, 0, 3, 1, 0])
n5 = len(defectos)


nivel_confianza5 = 0.95

# a)
media_muestral5 = np.mean(defectos)
print(f"a) Promedio de defectos: {media_muestral5:.4f}")

# b)
sigma5 = np.std(defectos, ddof=1)
print(f"b) Desviación estándar: {sigma5:.4f}")

# c)
error_estandar5 = sigma5 / np.sqrt(n5)
intervalo5 = st.norm.interval(confidence = nivel_confianza5, loc = media_muestral5, scale = error_estandar5)
intervalo5= (float(intervalo5[0]), float(intervalo5[1]))

print(f"\nc) Intervalo de confianza del 95% es: {intervalo5}")

print("\nd) Sí se recomienda una muestra mayor porque con n=30 el intervalo es bastante amplio.")
print("   Una muestra mayor reduciría el error estándar y proporcionaría una estimación más precisa.")


# $\color{indigo}{\text{EJEMPLO 6.}}$
# 
# Un artículo que apareció en el ejemplar de noviembre de 1983 de *Consumer Reports* comparó varios tipos de baterías.  
# Se informó que los siguientes datos provienen de una muestra de 20 baterías (en horas de duración):
# $$ [2200, 2290, 2390, 2410, 2480, 2500, 2580, 2700, 2030, 2100, 2190, 1600, 1740, 1900, 1930, 2000, 1510, 1470, 1770, 1710] $$
# Determine la media muestral, la desviación estándar muestral, y el intervalo de confianza del **90%** para la media poblacional.

# In[24]:


baterias = np.array([2200, 2290, 2390, 2410, 2480, 2500, 2580, 2700, 2030, 2100, 2190, 1600, 1740, 1900, 1930, 2000, 1510, 1470, 1770, 1710])
n6 = len(baterias)
media_muestral6 = np.mean(baterias)
sigma6 = np.std(baterias, ddof=1)
nivel_confianza6 = 0.90
error_estandar6 = sigma6 / np.sqrt(n6)

intervalo6 = st.norm.interval(confidence = nivel_confianza6, loc = media_muestral6, scale = error_estandar6)
intervalo6= (float(intervalo6[0]), float(intervalo6[1]))

print(f"Media muestral: {media_muestral6:.2f} ")
print(f"Desviación estándar: {sigma6:.2f}")
print(f"Intervalo de confianza del 90% es: {intervalo6}")


# In[25]:


# Datos
n3 = 250
media_muestral3 = 369
sigma3 = 50
nivel_confianza3 = 0.95
error_estandar3 = sigma3 / np.sqrt(n3)


intervalo3 = st.norm.interval(confidence = nivel_confianza3, loc = media_muestral3, scale = error_estandar3)
intervalo3= (float(intervalo3[0]), float(intervalo3[1]))
print(f"Intervalo de confianza del 95% es: {intervalo3}")


# $\color{indigo}{\text{EJEMPLO 7.}}$
# 
# El toxafen es un insecticida que ha sido identificado como contaminante en el ecosistema de los Grandes Lagos.  
# Para investigar el efecto de la exposición al toxafen en animales, a grupos de ratas se les administró toxafen en su dieta.
# 
# El artículo *"Reproduction Study of Toxaphene in Rat"* reporta **aumentos de peso (en gramos)** de ratas a las que se les administró una **dosis baja (4 ppm)** y de ratas de **control** cuya dieta no incluía el insecticida.
# 
# - En el grupo **control**, una muestra de **23 ratas hembras** tuvo una media de **$\bar{X_2} = 210$ g** y una desviación estándar de **32 g**.
# - En el grupo con **dosis baja**, una muestra de **20 ratas hembras** tuvo una media de **$\bar{X_1} = 190$ g** y una desviación estándar de **54 g**.
# 
# Encuentra un **intervalo de confianza del 90% para la diferencia de medias** poblacionales:  $\mu_1 - \mu_2$, donde:
# 
# - $\mu_1$ es la media poblacional de incremento de peso en el grupo con **dosis baja**.
# - $\mu_2$ es la media poblacional de incremento de peso en el grupo **control**.
# 
# ¿Sugiere este intervalo que el toxafen **reduce** el incremento de peso en ratas?  
# ¿Incluye el intervalo el valor 0? ¿Por qué es importante?

# In[26]:


# Grupo dosis baja
n7a = 20
media7a = 190
s7a = 54

# Grupo control
n7b = 23
media7b = 210
s7b = 32

confianza7 = 0.90

# Varianza agrupada
varianza_agrupada = ((n7a-1)*s7a**2 + (n7b-1)*s7b**2) / (n7a + n7b - 2)
error_estandar7 = np.sqrt(varianza_agrupada * (1/n7a + 1/n7b))
t_valor7 = st.t.ppf(1 - (1 - confianza7)/2, df=n7a+n7b-2)

diferencia = media7a - media7b
margen_error7 = t_valor7 * error_estandar7
ic7 = (diferencia - margen_error7, diferencia + margen_error7)


print(f"Intervalo de confianza del 90%: [{ic7[0]:.2f}, {ic7[1]:.2f}] \n")
print("El intervalo es completamente negativo, lo que sugiere que el toxafen reduce")
print("significativamente el aumento de peso. El intervalo no incluye el 0, lo que")
print("indica que la diferencia es estadísticamente significativa al nivel del 90%.")

