import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
taxa_acerto = []
taxa_acerto_completo = []
taxa_acerto_sem_coluna = []


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
    knn_completo.flit(X_train_completo, y_train_completo)

    # Previsões dos dados completos
    y_predicao_completo = knn_completo.predict(X_test_completo)
    taxa_acerto_completo = accuracy_score(y_test_completo, y_predicao_completo)
















