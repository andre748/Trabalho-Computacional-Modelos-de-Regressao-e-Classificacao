import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1. Carregamento e visualização dos dados
# =============================================

data = np.loadtxt("data/EMGsDataset.csv", delimiter=',')
data = data.T   # Transpor para (N, p+1)

X = data[:, :-1]  # Features (sensores)
Y = data[:, -1].astype(int)  # Labels (classes 1-5)

# Visualização inicial
colors = ['gray', 'teal', 'orange', 'purple', 'red']
labels = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']

plt.figure(figsize=(10, 6))
for i, classe in enumerate([1,2,3,4,5]):
    X_classe = X[Y == classe]
    plt.scatter(X_classe[:,0], X_classe[:,1], 
               c=colors[i], edgecolors='k', alpha=0.7,
               label=f'{labels[i]}', s=30)

plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.title("Dados EMG por Expressão Facial")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================
# 2. Funções auxiliares
# =============================================

def split(X, Y, p=0.8):
    """Divisão treino/teste aleatória"""
    n = len(X)
    idx = np.random.permutation(n)
    n_tr = int(p * n)
    return X[idx[:n_tr]], Y[idx[:n_tr]], X[idx[n_tr:]], Y[idx[n_tr:]]

def estat(v):
    """Estatísticas: média, desvio, max, min"""
    return np.mean(v), np.std(v), np.max(v), np.min(v)

# =============================================
# 3. Modelos de classificação
# =============================================

# --- MQO tradicional ---
def mqo_fit(X, Y):
    # Converter para one-hot encoding
    Y_onehot = np.zeros((len(Y), 5))
    for i in range(5):
        Y_onehot[Y == i+1, i] = 1
    
    # Adicionar intercepto
    X_aug = np.c_[np.ones(len(X)), X]
    return np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ Y_onehot

def mqo_predict(X, W):
    X_aug = np.c_[np.ones(len(X)), X]
    Y_pred = X_aug @ W
    return np.argmax(Y_pred, axis=1) + 1

# --- Naive Bayes ---
def naive_fit(X, Y):
    modelos = []
    for c in range(1, 6):
        X_c = X[Y == c]
        mu = X_c.mean(axis=0)
        var = X_c.var(axis=0) + 1e-6
        prior = np.log(len(X_c) / len(X))
        modelos.append((mu, var, prior))
    return modelos

def naive_predict(X, modelos):
    pred = []
    for x in X:
        scores = []
        for mu, var, prior in modelos:
            ll = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum((x - mu)**2 / var)
            scores.append(ll + prior)
        pred.append(np.argmax(scores) + 1)
    return np.array(pred)

# --- Gaussiano tradicional ---
def gauss_trad_fit(X, Y):
    mus, covs, priors = [], [], []
    for c in range(1, 6):
        X_c = X[Y == c]
        mus.append(X_c.mean(axis=0))
        covs.append(np.cov(X_c.T) + 1e-6 * np.eye(2))
        priors.append(np.log(len(X_c) / len(X)))
    return mus, covs, priors

# --- Gaussiano covariâncias iguais ---
def gauss_igual_fit(X, Y):
    mus, covs, priors = [], [], []
    cov_pooled = np.zeros((2, 2))
    
    for c in range(1, 6):
        X_c = X[Y == c]
        mus.append(X_c.mean(axis=0))
        priors.append(np.log(len(X_c) / len(X)))
        cov_pooled += (len(X_c) - 1) * np.cov(X_c.T)
    
    cov_pooled /= (len(X) - 5)
    cov_pooled += 1e-6 * np.eye(2)
    
    return mus, [cov_pooled] * 5, priors

# --- Gaussiano agregado ---
def gauss_agregado_fit(X, Y):
    mus, priors = [], []
    for c in range(1, 6):
        X_c = X[Y == c]
        mus.append(X_c.mean(axis=0))
        priors.append(np.log(len(X_c) / len(X)))
    
    cov_agregada = np.cov(X.T) + 1e-6 * np.eye(2)
    return mus, [cov_agregada] * 5, priors

# --- Gaussiano regularizado (Friedman) ---
def gauss_reg_fit(X, Y, lam):
    mus, covs, priors = [], [], []
    cov_pooled = np.cov(X.T)
    
    for c in range(1, 6):
        X_c = X[Y == c]
        mus.append(X_c.mean(axis=0))
        priors.append(np.log(len(X_c) / len(X)))
        
        cov_classe = np.cov(X_c.T)
        cov_reg = (1 - lam) * cov_classe + lam * cov_pooled
        covs.append(cov_reg + 1e-6 * np.eye(2))
    
    return mus, covs, priors

# --- Predição gaussiana (genérica) ---
def gauss_predict(X, modelo):
    mus, covs, priors = modelo
    pred = []
    
    for x in X:
        scores = []
        for mu, cov, prior in zip(mus, covs, priors):
            diff = x - mu
            inv_cov = np.linalg.pinv(cov)
            score = -0.5 * diff @ inv_cov @ diff - 0.5 * np.log(np.linalg.det(cov)) + prior
            scores.append(score)
        pred.append(np.argmax(scores) + 1)
    
    return np.array(pred)

# =============================================
# 4. Validação Monte Carlo
# =============================================

R = 500  # 500 rodadas conforme especificado
lambdas = [0.25, 0.5, 0.75]

modelos = {
    "MQO tradicional": (mqo_fit, mqo_predict),
    "Classificador Gaussiano Tradicional": (gauss_trad_fit, gauss_predict),
    "Classificador Gaussiano (Cov. de todo cj. treino)": (gauss_igual_fit, gauss_predict),
    "Classificador Gaussiano (Cov. Agregada)": (gauss_agregado_fit, gauss_predict),
    "Classificador de Bayes Ingênuo (Naive Bayes Classifier)": (naive_fit, naive_predict),
}

# Adicionar modelos regularizados
for lam in lambdas:
    nome = f"Classificador Gaussiano Regularizado (Friedman λ={lam})"
    modelos[nome] = (lambda X, Y, l=lam: gauss_reg_fit(X, Y, l), gauss_predict)

# Inicializar resultados
resultados = {nome: [] for nome in modelos}

print(f"Executando {R} rodadas Monte Carlo...")

# Executar simulação
for r in range(R):
    # Mostrar progresso de 2 em 2 rodadas
    if (r + 1) % 5 == 0:
        print(f"Rodada {r + 1}/{R}")
    
    X_tr, Y_tr, X_ts, Y_ts = split(X, Y)
    
    for nome, (fit_func, pred_func) in modelos.items():
        try:
            modelo = fit_func(X_tr, Y_tr)
            Y_pred = pred_func(X_ts, modelo)
            acuracia = np.mean(Y_pred == Y_ts)
            resultados[nome].append(acuracia)
        except:
            resultados[nome].append(0.0)  # Em caso de erro

# =============================================
# 5. Tabela de resultados
# =============================================

print("\n" + "="*80)
print("RESULTADOS FINAIS - CLASSIFICAÇÃO EMG")
print("="*80)
print(f"{'Modelo':<60} {'Média':>8} {'Desvio':>8} {'Menor':>8} {'Maior':>8}")
print("-" * 80)

for nome in modelos:
    if resultados[nome]:  # Se há resultados
        media, desvio, maior, menor = estat(resultados[nome])
        print(f"{nome:<60} {media:>8.4f} {desvio:>8.4f} {menor:>8.4f} {maior:>8.4f}")

# Identificar melhor modelo
melhor_nome = max(modelos.keys(), key=lambda x: np.mean(resultados[x]) if resultados[x] else 0)
melhor_acuracia = np.mean(resultados[melhor_nome])

print("-" * 80)
print(f"Melhor modelo: {melhor_nome}")
print(f"Acurácia média: {melhor_acuracia:.4f}")

# =============================================
# 6. Visualização dos resultados
# =============================================

plt.figure(figsize=(12, 8))

# Boxplot das acurácias
dados_plot = [resultados[nome] for nome in modelos if resultados[nome]]
nomes_plot = [nome.split('(')[0].strip() if '(' in nome else nome for nome in modelos if resultados[nome]]

plt.subplot(2, 1, 1)
plt.boxplot(dados_plot, labels=nomes_plot)
plt.ylabel('Acurácia')
plt.title('Distribuição das Acurácias por Modelo')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico de barras das médias
plt.subplot(2, 1, 2)
medias = [np.mean(dados) for dados in dados_plot]
plt.bar(range(len(nomes_plot)), medias, alpha=0.7)
plt.ylabel('Acurácia Média')
plt.xlabel('Modelos')
plt.title('Acurácia Média por Modelo')
plt.xticks(range(len(nomes_plot)), nomes_plot, rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== ANÁLISE DOS RESULTADOS ===")
print("1. Classes com sobreposição significativa (não linearmente separáveis)")
print("2. Modelos gaussianos geralmente superam MQO em classificação")
print("3. Regularização pode ajudar a controlar overfitting")
print("4. Dados EMG têm ruído natural, favorecendo modelos probabilísticos")