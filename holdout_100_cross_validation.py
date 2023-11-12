import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats


# Define as colunas do conjunto de dados
nome_coluna = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash', ' Magnesium',
' Total phenols', 'Flavanoids', 'Nonflavanoid phenols','Proanthocyanins',
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#Ler os dados
dados = pd.read_csv('wine.data', header=None, names= nome_coluna)
dados_ultima_coluna = dados.drop('Proline', axis=1)

print(dados)
print(dados_ultima_coluna)

holdout = 100

#Vetores para armazenar as diferentes taxa de acertos
lista_diferenca_taxa_acerto = []
lista_taxa_acerto_completo = []
lista_taxa_acerto_sem_coluna = []


for _ in range(holdout):
    #Divide os dados completos
    X_train_completo, X_test_completo, y_train_completo, y_test_completo = train_test_split(
        dados.iloc[:, 1:],
        dados['class'],
        test_size=0.5,
        random_state=42
    )
    # Divide os dados com uma coluna a menos
    X_train_incompleto, X_test_incompleto, y_train_incompleto, y_test_incompleto = train_test_split(
        dados_ultima_coluna.iloc[:, 1:],
        dados_ultima_coluna['class'],
        test_size=0.5,
        random_state=42
    )

    # Treina o classificador com todos os dados
    knn_completo = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_completo.fit(X_train_completo, y_train_completo)

    # Previsões dos dados completos
    y_predicao_completo = knn_completo.predict(X_test_completo)
    taxa_acerto_completo = accuracy_score(y_test_completo, y_predicao_completo)

    # Treina o classificador com todos os dados
    knn_incompleto = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_incompleto.fit(X_train_incompleto, y_train_incompleto)

    # Previsões dos dados completos
    y_predicao_incompleto = knn_incompleto.predict(X_test_incompleto)
    taxa_acerto_incompleto = accuracy_score(y_test_incompleto, y_predicao_incompleto)

    diferenca_taxa = taxa_acerto_completo - taxa_acerto_incompleto
    lista_diferenca_taxa_acerto.append(diferenca_taxa)

#Calcula a diferença
diferenca = sum(lista_diferenca_taxa_acerto) / len(lista_diferenca_taxa_acerto)

print(f'\n A média das diferenças')
print(diferenca)

intervalo_confianca = stats.t.interval(
    0.95,
    len(lista_diferenca_taxa_acerto) - 1,
    loc=np.mean(lista_diferenca_taxa_acerto),
    scale=stats.sem(lista_diferenca_taxa_acerto)
)

print(f'\n Intervalo de Confiança')
print(intervalo_confianca)

# Teste de Hipotese
estatistica, valor = stats.ttest_1samp(lista_diferenca_taxa_acerto, popmean=0)

# Nível de Significância
significancia = 0.05

if valor < significancia:
    print('Rejeita H0: A diferença da taxa de acerto é significativa entre as duas versões.')
else:
    print('Não Rejeita H0: Não há evidências suficientes para concluir que a diferença da taxa de acerto é significativa.')

# Intervalo de confiança da taxa acerto Dados Completos
intervalo_confiança_completo = stats.t.interval(
    0.95,
    len(y_test_completo) - 1,
    loc=taxa_acerto_completo,
    scale=stats.sem(y_test_completo)
)

print('\nIntervalo de confiança dos Dados completos')
print(intervalo_confiança_completo)

# Intervalo de confiança da taxa acerto Dados Inompletos
intervalo_confiança_incompleto = stats.t.interval(
    0.95,
    len(y_test_incompleto) - 1,
    loc=taxa_acerto_incompleto,
    scale=stats.sem(y_test_incompleto)
)

print('\nIntervalo de confiança dos Dados Incompletos')
print(intervalo_confiança_incompleto)
















