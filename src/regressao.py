import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, inv
import pandas as pd

def predicao(X, beta):
    return X @ beta

def mqo_regularizado(X, y, lambda_reg):
    """
    Implementação do MQO regularizado (Tikhonov)
    """
    p = X.shape[1]
    I = np.eye(p)
    # Para não regularizar o intercepto, colocamos 0 na primeira posição da diagonal
    I[0, 0] = 0
    
    beta_hat = inv(X.T @ X + lambda_reg * I) @ X.T @ y
    return beta_hat

# Carregar o arquivo .dat
data = np.loadtxt("data/aerogerador.dat")

controle_figura = True

# Visualização inicial dos dados
fig = plt.figure(1, figsize=(10, 6))
plt.scatter(data[:,0], data[:,1], edgecolor='k', alpha=0.7)
plt.title("Conjunto de dados do Aerogerador")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.grid(True, alpha=0.3)

# Organização dos dados
X_ = data[:,0].reshape(-1, 1)  # Variável independente (velocidade do vento)
N, p = X_.shape
y = data[:,1].reshape(-1, 1)   # Variável dependente (potência gerada)

# Adicionar coluna de 1s para o intercepto
X = np.hstack((np.ones((N,1)), X_))

# Parâmetros da simulação
rodadas = 500
lambdas = [0, 0.25, 0.5, 0.75, 1.0]

# Listas para armazenar RSS de cada modelo
rss_media = []
rss_mqo_tradicional = []
rss_mqo_regularizado = {lam: [] for lam in lambdas}

print("Iniciando simulação Monte Carlo com 500 rodadas...")

for r in range(rodadas):
    if (r + 1) % 100 == 0:
        print(f"Rodada {r + 1}/500")
    
    # Embaralhar o conjunto de dados
    idx = np.random.permutation(N)
    X_embaralhado = X[idx,:]
    y_embaralhado = y[idx,:]
    
    # Particionamento 80% treino / 20% teste
    split_idx = int(N * 0.8)
    X_treino = X_embaralhado[:split_idx,:]
    y_treino = y_embaralhado[:split_idx,:]
    X_teste = X_embaralhado[split_idx:,:]
    y_teste = y_embaralhado[split_idx:,:]
    
    # 1. Modelo baseado na média
    media_y_treino = np.mean(y_treino)
    beta_media = np.array([[media_y_treino], [0]])
    y_pred_media = predicao(X_teste, beta_media)
    rss_media.append(np.sum((y_teste - y_pred_media)**2))
    
    # 2. MQO tradicional
    beta_mqo = inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred_mqo = predicao(X_teste, beta_mqo)
    rss_mqo_tradicional.append(np.sum((y_teste - y_pred_mqo)**2))
    
    # 3. MQO regularizado para diferentes valores de lambda
    for lam in lambdas:
        beta_reg = mqo_regularizado(X_treino, y_treino, lam)
        y_pred_reg = predicao(X_teste, beta_reg)
        rss_mqo_regularizado[lam].append(np.sum((y_teste - y_pred_reg)**2))
    
    # Visualização apenas na primeira rodada
    if controle_figura:
        fig = plt.figure(2, figsize=(15, 10))
        
        # Subplot 1: Dados de treino com modelos ajustados
        plt.subplot(2, 2, 1)
        plt.scatter(X_treino[:,1], y_treino[:,0], edgecolor='k', alpha=0.7, label='Dados de treino')
        
        # Criar linha para visualização dos modelos
        x_range = np.linspace(np.min(X_treino[:,1]), np.max(X_treino[:,1]), 100)
        X_range = np.column_stack([np.ones(100), x_range])
        
        # Plotar diferentes modelos
        y_pred_media_plot = predicao(X_range, beta_media)
        y_pred_mqo_plot = predicao(X_range, beta_mqo)
        
        plt.plot(x_range, y_pred_media_plot, 'g:', linewidth=2, label='Modelo Média')
        plt.plot(x_range, y_pred_mqo_plot, 'r-', linewidth=2, label='MQO Tradicional')
        
        # Plotar alguns modelos regularizados
        for lam in [0, 0.5, 1.0]:
            beta_reg_plot = mqo_regularizado(X_treino, y_treino, lam)
            y_pred_reg_plot = predicao(X_range, beta_reg_plot)
            plt.plot(x_range, y_pred_reg_plot, '--', linewidth=2, label=f'MQO Reg. (λ={lam})')
        
        plt.xlabel("Velocidade do Vento")
        plt.ylabel("Potência Gerada")
        plt.title("Modelos Ajustados - Dados de Treino")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Dados de teste
        plt.subplot(2, 2, 2)
        plt.scatter(X_teste[:,1], y_teste[:,0], edgecolor='k', alpha=0.7, color='orange')
        plt.xlabel("Velocidade do Vento")
        plt.ylabel("Potência Gerada")
        plt.title("Dados de Teste")
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Resíduos do MQO tradicional
        plt.subplot(2, 2, 3)
        residuos = y_treino - predicao(X_treino, beta_mqo)
        plt.hist(residuos.flatten(), bins=20, edgecolor='k', alpha=0.7)
        plt.xlabel("Resíduos")
        plt.ylabel("Frequência")
        plt.title("Histograma dos Resíduos - MQO Tradicional")
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Comparação de RSS na primeira rodada
        plt.subplot(2, 2, 4)
        modelos = ['Média', 'MQO'] + [f'MQO λ={lam}' for lam in lambdas]
        rss_primeira_rodada = [rss_media[0], rss_mqo_tradicional[0]] + [rss_mqo_regularizado[lam][0] for lam in lambdas]
        plt.bar(range(len(modelos)), rss_primeira_rodada)
        plt.xlabel("Modelos")
        plt.ylabel("RSS")
        plt.title("RSS - Primeira Rodada")
        plt.xticks(range(len(modelos)), modelos, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        controle_figura = False

# Calcular estatísticas finais
print("\n" + "="*80)
print("RESULTADOS FINAIS - TAREFA DE REGRESSÃO")
print("="*80)

# Criar tabela de resultados
resultados = []

# Modelo Média
stats_media = {
    'Modelo': 'Média da variável dependente',
    'Média': np.mean(rss_media),
    'Desvio-Padrão': np.std(rss_media),
    'Maior Valor': np.max(rss_media),
    'Menor Valor': np.min(rss_media)
}
resultados.append(stats_media)

# MQO tradicional
stats_mqo = {
    'Modelo': 'MQO tradicional',
    'Média': np.mean(rss_mqo_tradicional),
    'Desvio-Padrão': np.std(rss_mqo_tradicional),
    'Maior Valor': np.max(rss_mqo_tradicional),
    'Menor Valor': np.min(rss_mqo_tradicional)
}
resultados.append(stats_mqo)

# MQO regularizados
for lam in lambdas:
    stats_reg = {
        'Modelo': f'MQO regularizado ({lam})',
        'Média': np.mean(rss_mqo_regularizado[lam]),
        'Desvio-Padrão': np.std(rss_mqo_regularizado[lam]),
        'Maior Valor': np.max(rss_mqo_regularizado[lam]),
        'Menor Valor': np.min(rss_mqo_regularizado[lam])
    }
    resultados.append(stats_reg)

# Criar DataFrame e exibir tabela
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False, float_format='%.4f'))

# Gráfico de comparação dos modelos
plt.figure(3, figsize=(12, 8))

# Subplot 1: Boxplot dos RSS
plt.subplot(2, 2, 1)
dados_boxplot = [rss_media, rss_mqo_tradicional] + [rss_mqo_regularizado[lam] for lam in lambdas]
labels_boxplot = ['Média', 'MQO'] + [f'MQO λ={lam}' for lam in lambdas]
plt.boxplot(dados_boxplot, labels=labels_boxplot)
plt.ylabel('RSS')
plt.title('Distribuição dos RSS por Modelo')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Subplot 2: Médias dos RSS
plt.subplot(2, 2, 2)
medias_rss = [np.mean(rss_media), np.mean(rss_mqo_tradicional)] + [np.mean(rss_mqo_regularizado[lam]) for lam in lambdas]
plt.bar(range(len(labels_boxplot)), medias_rss, alpha=0.7)
plt.xlabel('Modelos')
plt.ylabel('RSS Médio')
plt.title('RSS Médio por Modelo')
plt.xticks(range(len(labels_boxplot)), labels_boxplot, rotation=45)
plt.grid(True, alpha=0.3)

# Subplot 3: Desvio padrão dos RSS
plt.subplot(2, 2, 3)
desvios_rss = [np.std(rss_media), np.std(rss_mqo_tradicional)] + [np.std(rss_mqo_regularizado[lam]) for lam in lambdas]
plt.bar(range(len(labels_boxplot)), desvios_rss, alpha=0.7, color='orange')
plt.xlabel('Modelos')
plt.ylabel('Desvio Padrão RSS')
plt.title('Variabilidade dos RSS por Modelo')
plt.xticks(range(len(labels_boxplot)), labels_boxplot, rotation=45)
plt.grid(True, alpha=0.3)

# Subplot 4: Evolução do RSS ao longo das rodadas (primeiras 50 rodadas)
plt.subplot(2, 2, 4)
plt.plot(rss_mqo_tradicional[:50], label='MQO Tradicional', alpha=0.8)
plt.plot(rss_mqo_regularizado[0.5][:50], label='MQO λ=0.5', alpha=0.8)
plt.plot(rss_media[:50], label='Modelo Média', alpha=0.8)
plt.xlabel('Rodadas')
plt.ylabel('RSS')
plt.title('Evolução do RSS (primeiras 50 rodadas)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

print(f"\n\nMelhor modelo (menor RSS médio): {df_resultados.loc[df_resultados['Média'].idxmin(), 'Modelo']}")
print(f"RSS médio do melhor modelo: {df_resultados['Média'].min():.4f}")

plt.show()