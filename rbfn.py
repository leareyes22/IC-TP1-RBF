# Importar librerías
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Crear problema
p = 50 # Número de muestras
x = np.linspace(-5, 5, p).reshape(1,-1)
y = 2 * np.cos(x) + np.sin(3*x) + 5

# Dibujar puntos
plt.plot(x,y, 'or')


# Definir número de neuronas
k = 20

# Agrupar puntos en clústers
model = KMeans(n_clusters=k)
model.fit(x.T)

# Extraer centroides
c = model.cluster_centers_

# Calcular el sigma
sigma = (max(c)-min(c))/np.sqrt(2*k)
sigma = sigma[0]

# Calcular matriz G
G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        dist = np.linalg.norm(x[0,i]-c[j], 2)
        G[i,j] = np.exp((-1/(sigma**2))*dist**2)

W = np.dot(np.linalg.pinv(G), y.T)

# Propagar la red
p = 200
xnew = x = np.linspace(-5, 5, p).reshape(1,-1)

G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        dist = np.linalg.norm(x[0,i]-c[j], 2)
        G[i,j] = np.exp((-1/(sigma**2))*dist**2)

ynew = np.dot(G, W)
# Dibujar puntos
plt.plot(xnew.T, ynew, '-b')
plt.show()