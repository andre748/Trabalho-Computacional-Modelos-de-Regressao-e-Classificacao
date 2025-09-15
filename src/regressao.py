import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def predicao(X, beta):
    return X @ beta

# Carregar o arquivo .dat
data = np.loadtxt("data/aerogerador.dat")

controle_figura = True

# ALTERAÇÃO: Gráfico 2D em vez de 3D (pois temos apenas 2 colunas)
fig = plt.figure(1)
plt.scatter(data[:,0], data[:,1], edgecolor='k')
plt.title("Todo o conjunto de dados EMG")
plt.xlabel("EMG Canal 1 (Input)")
plt.ylabel("Força/Tensão (Output)")

# Dados sem a coluna de 1s (X_)
X_ = data[:,0].reshape(-1, 1)  # Apenas a primeira coluna como feature
N, p = X_.shape
y = data[:,1].reshape(-1, 1)   # Segunda coluna como target

# Dados com a coluna de 1s (X)
X = np.hstack((
    np.ones((N,1)), X_
))

rodadas = 1000

# Arrays para armazenar métricas
r2_mqo_list = []
r2_mqo_sem_intercepto_list = []
r2_media_list = []

for r in range(rodadas):
    # EMBARALHAR O CONJUNTO DE DADOS
    idx = np.random.permutation(N)
    Xr_ = X_[idx,:]
    Xr = X[idx,:]
    yr = y[idx,:]

    # Particionamento do conjunto de dados (80/20)
    X_treino = Xr[:int(N*.8),:]
    X_treino_ = Xr_[:int(N*.8),:]
    y_treino = yr[:int(N*.8),:]

    X_teste = Xr[int(N*.8):,:]
    X_teste_ = Xr_[int(N*.8):,:]
    y_teste = yr[int(N*.8):,:]

    # Treinamento dos modelos:
    # Modelo baseado na média:
    beta_hat_media = np.array([
        [np.mean(y_treino)],
        [0]
    ])

    # Modelo baseado MQO (sem intercepto)
    beta_hat_ = pinv(X_treino_.T @ X_treino_) @ X_treino_.T @ y_treino
    beta_hat_ = np.vstack((
        np.zeros((1,1)), beta_hat_
    ))

    # Modelo MQO tradicional
    beta_hat = pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino    

    if controle_figura:
        # ALTERAÇÃO: Gráficos 2D para regressão linear simples
        fig = plt.figure(2)
        plt.subplot(1,2,1)
        plt.scatter(X_treino[:,1], y_treino[:,0], edgecolor='k', alpha=0.7)
        plt.xlabel("EMG Canal 1")
        plt.ylabel("Força/Tensão")
        plt.title("Observações de Treino EMG")
        
        # Plotar linha de regressão
        x_range = np.linspace(np.min(X_treino[:,1]), np.max(X_treino[:,1]), 100)
        X_range = np.column_stack([np.ones(100), x_range.reshape(-1,1)])
        
        y_pred_mqo = predicao(X_range, beta_hat)
        y_pred_sem_intercepto = predicao(X_range, beta_hat_)
        y_pred_media = predicao(X_range, beta_hat_media)
        
        plt.plot(x_range, y_pred_mqo, 'r-', linewidth=2, label='MQO com intercepto')
        plt.plot(x_range, y_pred_sem_intercepto, 'b--', linewidth=2, label='MQO sem intercepto')
        plt.plot(x_range, y_pred_media, 'g:', linewidth=2, label='Modelo média')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1,2,2)
        plt.scatter(X_teste[:,1], y_teste[:,0], edgecolor='k', alpha=0.7)
        plt.xlabel("EMG Canal 1")
        plt.ylabel("Força/Tensão")
        plt.title("Observações de Teste EMG")
        plt.grid(True, alpha=0.3)

        # Cálculo de epsilon
        y_pred = predicao(X_treino, beta_hat)
        desvios = y_pred - y_treino

        plt.figure(3)
        plt.hist(desvios, edgecolor='k', color='green', bins=20)
        plt.title(r"Histograma de desvios ($\varepsilon$)")
        controle_figura = False

    # Teste de desempenho e armazenamento de métricas:
    # Modelo MQO (com intercepto)
    y_pred = predicao(X_teste, beta_hat)
    desvios = y_teste - y_pred
    SSE = np.sum(desvios**2)
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    r2_mqo_list.append(R2)
    
    # Modelo MQO (sem intercepto)
    y_pred = predicao(X_teste, beta_hat_)
    desvios = y_teste - y_pred
    SSE = np.sum(desvios**2)
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    r2_mqo_sem_intercepto_list.append(R2)
    
    # Modelo média
    y_pred = predicao(X_teste, beta_hat_media)
    desvios = y_teste - y_pred
    SSE = np.sum(desvios**2)
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    r2_media_list.append(R2)

# Calcular estatísticas finais
print("\n=== RESULTADOS FINAIS ===")
print(f"MQO com intercepto - R² médio: {np.mean(r2_mqo_list):.4f} ± {np.std(r2_mqo_list):.4f}")
print(f"MQO sem intercepto - R² médio: {np.mean(r2_mqo_sem_intercepto_list):.4f} ± {np.std(r2_mqo_sem_intercepto_list):.4f}")
print(f"Modelo média - R² médio: {np.mean(r2_media_list):.4f} ± {np.std(r2_media_list):.4f}")

plt.show()