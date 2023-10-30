import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

nome_colunas = ['sepal lenght', 'sepal width', 'petal lenght', 'petal width', 'class']

# Metade de cada classe como teste
dados_teste = pd.read_csv('iris_data_teste.data', header=None, names=nome_colunas)
print("Dados de Teste")
print(dados_teste)


# Metade de cada classe como treinamento
dados_treinamento = pd.read_csv('iris_data_treinamento.data', header=None, names=nome_colunas)
print("Dados de Treinamento")
print(dados_treinamento)

knn = KNeighborsClassifier(n_neighbors=1)

# Faz previsão
knn.fit(dados_treinamento[nome_colunas[:-1]], dados_treinamento['class'])
knn.predict(dados_teste[nome_colunas[:-1]])

# Taxa de acerto
acertos = knn.score(dados_teste[nome_colunas[:-1]], dados_teste['class'])

print(acertos)