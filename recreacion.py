import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Carga de datos y creando sus indices para pasarlos (dato,indice)
train_input = np.asarray(pd.read_csv('train_data_input.csv', sep=',', header=None))
_,train_input_index = np.meshgrid(np.linspace(1,8,8),np.linspace(1,9,9))

train_output = np.asarray(pd.read_csv('train_data_output.csv', sep=',', header=None))
_,train_output_index = np.meshgrid(np.linspace(1,3,3),np.linspace(1,9,9))

test_input = np.asarray(pd.read_csv('test_data_input.csv', sep=',', header=None))
_,test_input_index = np.meshgrid(np.linspace(1,8,8),np.linspace(1,3,3))

# Definir número de neuronas
k = 25

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
        dist = np.linalg.norm(x[0,i]-c[j], 2) # Distancia euclideana Entre Xi y Cj
        G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

W = np.dot(np.linalg.pinv(G), y.T)

# Propagar la red
p = 200
xnew = x = np.linspace(-5, 5, p).reshape(1,-1)

G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        dist = np.linalg.norm(x[0,i]-c[j], 2) # Distancia euclideana Entre Xi y Cj
        G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

ynew = np.dot(G, W) # Salida de la red
# Dibujar puntos
plt.plot(xnew.T, ynew, '-b')
plt.show()
