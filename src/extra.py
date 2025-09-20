import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd

def predicao(X, beta):
    return X @ beta

def mqo_regularizado(X, y, lambda_reg):
    """MQO regularizado (Tikhonov)"""
    p = X.shape[1]
    I = np.eye(p)
    I[0, 0] = 0  # Não regularizar intercepto
    beta_hat = inv(X.T @ X + lambda_reg * I) @ X.T @ y
    return beta_hat

def criar_features_polinomiais(X, grau):
    """
    Criar features polinomiais até o grau especificado
    X: matriz (N, 1) com feature original
    grau: grau máximo do polinômio
    Retorna: matriz (N, grau+1) com [1, x, x^2, ..., x^grau]
    """
    N = X.shape[0]
    X_poly = np.ones((N, 1))  # Intercepto
    
    for i in range(1, grau + 1):
        X_poly = np.hstack([X_poly, X**i])
    
    return X_poly

def mqo_polinomial(X, y, grau):
    """
    Regressão polinomial usando MQO
    """
    X_poly = criar_features_polinomiais(X, grau)
    beta_hat = inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return beta_hat, X_poly

# Carregar dados
data = np.loadtxt("data/aerogerador.dat")

# Separar variáveis
X_original = data[:, 0].reshape(-1, 1)  # Velocidade do vento
N, p_original = X_original.shape
y = data[:, 1].reshape(-1, 1)  # Potência gerada

print("=== QUESTÃO EXTRA - REGRESSÃO POLINOMIAL ===")
print(f"Dados carregados: {N} amostras")

# Análise exploratória da não-linearidade
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_original, y, alpha=0.6, edgecolors='k')
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Dados Originais - Relação Não-Linear Evidente")
plt.grid(True, alpha=0.3)

# Testar diferentes graus de polinômio
graus_teste = [1, 2, 3, 4, 5]
cores_poly = ['red', 'blue', 'green', 'orange', 'purple']

# Ajustar polinômios para visualização
x_plot = np.linspace(X_original.min(), X_original.max(), 300).reshape(-1, 1)

plt.subplot(2, 2, 2)
plt.scatter(X_original, y, alpha=0.4, edgecolors='k', s=20)

for i, grau in enumerate(graus_teste):
    # Treinar com todos os dados para visualização
    beta_poly, _ = mqo_polinomial(X_original, y, grau)
    X_plot_poly = criar_features_polinomiais(x_plot, grau)
    y_plot_pred = X_plot_poly @ beta_poly
    
    plt.plot(x_plot, y_plot_pred, color=cores_poly[i], 
             linewidth=2, label=f'Grau {grau}')

plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Comparação de Diferentes Graus Polinomiais")
plt.legend()
plt.grid(True, alpha=0.3)

# Analisar ajuste por grau (R² no conjunto completo)
plt.subplot(2, 2, 3)
r2_por_grau = []

for grau in graus_teste:
    beta_poly, X_poly = mqo_polinomial(X_original, y, grau)
    y_pred = X_poly @ beta_poly
    
    # Calcular R²
    SSE = np.sum((y - y_pred)**2)
    SST = np.sum((y - np.mean(y))**2)
    r2 = 1 - SSE/SST
    r2_por_grau.append(r2)

plt.plot(graus_teste, r2_por_grau, 'o-', linewidth=2, markersize=8)
plt.xlabel("Grau do Polinômio")
plt.ylabel("R² (Conjunto Completo)")
plt.title("R² vs Grau do Polinômio")
plt.grid(True, alpha=0.3)
plt.xticks(graus_teste)

# Mostrar R² por grau
print(f"\nR² por grau (conjunto completo):")
for grau, r2 in zip(graus_teste, r2_por_grau):
    print(f"Grau {grau}: R² = {r2:.4f}")

# Escolher grau baseado no R² e interpretabilidade
grau_escolhido = 3  # Cubico - bom compromisso entre ajuste e complexidade
print(f"\nGrau escolhido para validação Monte Carlo: {grau_escolhido}")
print("Justificativa: Grau 3 captura a não-linearidade sem overfitting excessivo")

# Análise residual do grau escolhido
beta_escolhido, X_poly_escolhido = mqo_polinomial(X_original, y, grau_escolhido)
y_pred_escolhido = X_poly_escolhido @ beta_escolhido
residuos = y - y_pred_escolhido

plt.subplot(2, 2, 4)
plt.scatter(y_pred_escolhido, residuos, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Valores Preditos")
plt.ylabel("Resíduos")
plt.title(f"Análise de Resíduos - Polinômio Grau {grau_escolhido}")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# VALIDAÇÃO MONTE CARLO COM MODELO POLINOMIAL
# ==========================================

print(f"\nIniciando validação Monte Carlo com modelo polinomial (grau {grau_escolhido})...")

rodadas = 500
lambdas = [0, 0.25, 0.5, 0.75, 1.0]

# Listas para armazenar RSS
rss_media = []
rss_mqo_tradicional = []
rss_mqo_regularizado = {lam: [] for lam in lambdas}
rss_mqo_polinomial = []  # NOVA LISTA PARA MODELO POLINOMIAL

for r in range(rodadas):
    if (r + 1) % 100 == 0:
        print(f"Rodada {r + 1}/500")
    
    # Embaralhar dados
    idx = np.random.permutation(N)
    X_embaralhado = X_original[idx]
    y_embaralhado = y[idx]
    
    # Particionamento 80/20
    split_idx = int(N * 0.8)
    X_treino = X_embaralhado[:split_idx]
    y_treino = y_embaralhado[:split_idx]
    X_teste = X_embaralhado[split_idx:]
    y_teste = y_embaralhado[split_idx:]
    
    # Adicionar intercepto para modelos lineares
    X_treino_linear = np.hstack([np.ones((len(X_treino), 1)), X_treino])
    X_teste_linear = np.hstack([np.ones((len(X_teste), 1)), X_teste])
    
    # 1. Modelo média
    media_y_treino = np.mean(y_treino)
    beta_media = np.array([[media_y_treino], [0]])
    y_pred_media = X_teste_linear @ beta_media
    rss_media.append(np.sum((y_teste - y_pred_media)**2))
    
    # 2. MQO tradicional (linear)
    beta_linear = inv(X_treino_linear.T @ X_treino_linear) @ X_treino_linear.T @ y_treino
    y_pred_linear = X_teste_linear @ beta_linear
    rss_mqo_tradicional.append(np.sum((y_teste - y_pred_linear)**2))
    
    # 3. MQO regularizado (linear)
    for lam in lambdas:
        beta_reg = mqo_regularizado(X_treino_linear, y_treino, lam)
        y_pred_reg = X_teste_linear @ beta_reg
        rss_mqo_regularizado[lam].append(np.sum((y_teste - y_pred_reg)**2))
    
    # 4. MQO POLINOMIAL (NOVO MODELO)
    # Treinar modelo polinomial
    X_treino_poly = criar_features_polinomiais(X_treino, grau_escolhido)
    X_teste_poly = criar_features_polinomiais(X_teste, grau_escolhido)
    
    try:
        beta_poly = inv(X_treino_poly.T @ X_treino_poly) @ X_treino_poly.T @ y_treino
        y_pred_poly = X_teste_poly @ beta_poly
        rss_poly = np.sum((y_teste - y_pred_poly)**2)
        rss_mqo_polinomial.append(rss_poly)
    except:
        # Em caso de problemas numéricos
        rss_mqo_polinomial.append(float('inf'))

# ==========================================
# TABELA DE RESULTADOS EXPANDIDA
# ==========================================

print("\n" + "="*80)
print("RESULTADOS FINAIS - REGRESSÃO COM MODELO POLINOMIAL")
print("="*80)

resultados = []

# Modelos existentes
modelos_dados = [
    ('Média da variável dependente', rss_media),
    ('MQO tradicional (linear)', rss_mqo_tradicional),
    (f'MQO Polinomial (grau {grau_escolhido})', rss_mqo_polinomial),  # NOVO
]

# Adicionar MQO regularizados
for lam in lambdas:
    modelos_dados.append((f'MQO regularizado ({lam})', rss_mqo_regularizado[lam]))

# Calcular estatísticas
for nome, lista_rss in modelos_dados:
    # Filtrar valores infinitos
    lista_valida = [x for x in lista_rss if not np.isinf(x)]
    
    if lista_valida:
        resultado = {
            'Modelo': nome,
            'Média': np.mean(lista_valida),
            'Desvio-Padrão': np.std(lista_valida),
            'Maior Valor': np.max(lista_valida),
            'Menor Valor': np.min(lista_valida)
        }
        resultados.append(resultado)

# Criar e exibir tabela
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False, float_format='%.4f'))

# Identificar melhor modelo
melhor_idx = df_resultados['Média'].idxmin()  # Menor RSS é melhor
melhor_modelo = df_resultados.loc[melhor_idx, 'Modelo']
melhor_rss = df_resultados.loc[melhor_idx, 'Média']

print(f"\n\nMelhor modelo: {melhor_modelo}")
print(f"RSS médio: {melhor_rss:.4f}")

# ==========================================
# ANÁLISE E DISCUSSÃO
# ==========================================

print(f"\n=== DISCUSSÃO DOS RESULTADOS ===")

# Comparar modelo linear vs polinomial
rss_linear_medio = np.mean(rss_mqo_tradicional)
rss_poly_medio = np.mean([x for x in rss_mqo_polinomial if not np.isinf(x)])

melhoria_percentual = ((rss_linear_medio - rss_poly_medio) / rss_linear_medio) * 100

print(f"Comparação MQO Linear vs Polinomial:")
print(f"  - MQO Linear (grau 1): RSS médio = {rss_linear_medio:.4f}")
print(f"  - MQO Polinomial (grau {grau_escolhido}): RSS médio = {rss_poly_medio:.4f}")
print(f"  - Melhoria: {melhoria_percentual:.2f}%")

if melhoria_percentual > 5:
    print(f"✓ Modelo polinomial apresenta melhoria significativa!")
elif melhoria_percentual > 0:
    print(f"→ Modelo polinomial apresenta melhoria modesta")
else:
    print(f"✗ Modelo polinomial não melhora significativamente")

print(f"\nJustificativa técnica:")
print(f"1. Dados do aerogerador apresentam relação não-linear evidente")
print(f"2. Curva característica: crescimento lento → rápido → saturação")
print(f"3. Polinômio grau {grau_escolhido} captura essa não-linearidade")
print(f"4. Modelo linear é limitado para capturar essa complexidade")

# Visualização comparativa final
plt.figure(figsize=(12, 6))

# Gráfico 1: Comparação RSS
plt.subplot(1, 2, 1)
modelos_comparacao = ['Média', 'Linear', f'Polinomial\n(grau {grau_escolhido})', 'Regularizado\n(λ=0.5)']
rss_comparacao = [
    np.mean(rss_media),
    rss_linear_medio,
    rss_poly_medio,
    np.mean(rss_mqo_regularizado[0.5])
]

colors = ['red', 'blue', 'green', 'orange']
bars = plt.bar(modelos_comparacao, rss_comparacao, color=colors, alpha=0.7)

# Destacar o melhor
melhor_pos = np.argmin(rss_comparacao)
bars[melhor_pos].set_color('gold')
bars[melhor_pos].set_edgecolor('black')
bars[melhor_pos].set_linewidth(2)

plt.ylabel('RSS Médio')
plt.title('Comparação de Performance dos Modelos')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 2: Ajuste visual do melhor modelo
plt.subplot(1, 2, 2)
plt.scatter(X_original, y, alpha=0.4, s=20, label='Dados reais')

# Plotar modelo linear
x_plot = np.linspace(X_original.min(), X_original.max(), 300).reshape(-1, 1)
X_plot_linear = np.hstack([np.ones((300, 1)), x_plot])
beta_linear_final, _ = mqo_polinomial(X_original, y, 1)  # Grau 1 = linear
X_linear_final = criar_features_polinomiais(X_original, 1)
beta_linear_final = inv(X_linear_final.T @ X_linear_final) @ X_linear_final.T @ y
X_plot_linear_final = criar_features_polinomiais(x_plot, 1)
y_plot_linear = X_plot_linear_final @ beta_linear_final

# Plotar modelo polinomial
beta_poly_final, _ = mqo_polinomial(X_original, y, grau_escolhido)
X_plot_poly_final = criar_features_polinomiais(x_plot, grau_escolhido)
y_plot_poly = X_plot_poly_final @ beta_poly_final

plt.plot(x_plot, y_plot_linear, 'b--', linewidth=2, label='Modelo Linear')
plt.plot(x_plot, y_plot_poly, 'g-', linewidth=3, label=f'Modelo Polinomial (grau {grau_escolhido})')

plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Comparação Visual: Linear vs Polinomial')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== CONCLUSÃO DA QUESTÃO EXTRA ===")
print(f"O modelo polinomial de grau {grau_escolhido} {'melhora significativamente' if melhoria_percentual > 5 else 'apresenta' if melhoria_percentual > 0 else 'não melhora'} a performance")
print(f"em relação aos modelos lineares, demonstrando a importância de considerar")
print(f"a não-linearidade inerente aos dados de aerogeradores.")