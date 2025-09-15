import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Carregar dados
data = np.loadtxt("data/EMGsDataset.csv", delimiter=',')
data = data.T   # Transpor para (N, p+1)

N, p = data.shape
p -= 1

# Todas as classes (1 a 5)
classes = [1,2,3,4,5]
nomes = ["Neutro", "Sorriso", "Sobrancelhas levantadas", "Surpreso", "Rabugento"]
cores = ["gray", "teal", "orange", "purple", "red"]

C = len(classes)

# Construir X e Y
X = np.empty((0,p))
Y = np.empty((0,C))

for i,classe in enumerate(classes):
    X_classe = data[data[:,-1]==classe,:-1]
    y_rotulo = -np.ones((1,C))
    y_rotulo[0,i] = 1
    Y = np.vstack(( Y, np.tile(y_rotulo,(X_classe.shape[0],1)) ))
    X = np.vstack(( X, X_classe ))

    plt.scatter(X_classe[:,0],X_classe[:,1],
                c=cores[i],edgecolors='k',
                label=nomes[i])

plt.legend()
plt.xlabel("Sensor 1 (Corrugador do Supercílio)",fontsize=12)
plt.ylabel("Sensor 2 (Zigomático Maior)",fontsize=12)
plt.title("Conjunto de dados EMG")

# Adicionar coluna de 1's
N = X.shape[0]
X = np.hstack(( np.ones((N,1)), X ))

# Treinar MQO
W_hat = inv(X.T@X) @ X.T @ Y

# Plotar fronteiras de decisão
x1 = np.linspace(-100, 4200, 300)
X1,X2 = np.meshgrid(x1,x1)

X3d = np.concatenate((
    np.ones((300,300,1)),
    X1.reshape(300,300,1),
    X2.reshape(300,300,1),
),axis=2)

Y_pred = np.argmax(X3d@W_hat,axis=2)
plt.contourf(X1,X2,Y_pred,alpha=.3,levels=np.arange(C+1)-0.5,cmap="tab10")

plt.xlim(-100,4095)
plt.ylim(-100,4095)

plt.show()
