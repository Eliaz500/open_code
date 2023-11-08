####################################################################
# Realize um Holdout com 1-NN (distância Euclidiana),
# utilizando 70% dos dados para treinamento e o restante (30%)
# para teste na base de dados
# Wine archive.ics.uci.edu/ml/datasets/Wine.
# Você precisa mostrar como calculou cada métrica,
# não pode utilizar biblioteca que já calcula a métrica
# diretamente mas pode utilizar biblioteca para o 1-NN
# e para dividir os dados entre treino e teste.
#####################################################################



import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define as colunas do conjunto de dados
name = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash', ' Magnesium',
' Total phenols', 'Flavanoids', 'Nonflavanoid phenols','Proanthocyanins',
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


# Ler o conjunto de dados
dados = pd.read_csv('wine.data', header=None, names= name)

print(dados)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(dados[name[1:]], dados['class'], test_size=0.3, random_state=5)

print('\nX Treinamento')
print(X_train)
print('\nX Teste')
print(X_test)
print('\ny Treinamento')
print(y_train)
print('\ny Teste')
print(y_test)



# Instanciação dos classificadores KNN, com os 1 vizinhos mais próximos
knn = KNeighborsClassifier(n_neighbors=1)

# Treinamento dos classificadores
knn.fit(X_train, y_train)


# Previsão dos conjuntos
predicao = knn.predict(X_test)

# Calcular a acurácia manualmente
acuracia = np.mean(y_test == predicao)


# Imprime as taxas de acerto
print('1NN - Taxa de acerto: ', acuracia)

# Determinar o número de classes
numero_classes = len(np.unique(np.concatenate((y_train, y_test))))


# Inicializar a matriz de confusão
matriz_confusao = np.zeros((numero_classes, numero_classes), dtype=int)


# Calcular a matriz de confusão
for verdadeiro, previsto in zip(y_test, predicao):
    matriz_confusao[verdadeiro - 1][previsto - 1] += 1
    #Imprime o Matriz de Confusão
print( '\nMatriz de confusão \n',matriz_confusao)


# Calcular a precisão manualmente
# A funcao diag() retorna os elementos da diagonal principal da matriz de confusão
# A funcao sum() realiza a soma dos elementos da matriz de confusão ao longo do eixo 0
precisao = np.diag(matriz_confusao) / np.sum(matriz_confusao, axis=0)
#Imprime o Precisão
print( '\nPrecisão \n',precisao)


# Calcular o recall manualmente
# Calcular a precisão manualmente
# A funcao diag() retorna os elementos da diagonal principal da matriz de confusão
# A funcao sum() realiza a soma dos elementos da matriz de confusão ao longo do eixo 1
recall = np.diag(matriz_confusao) / np.sum(matriz_confusao, axis=1)
#Imprime o Recall
print( '\nRecall \n',recall)


# Calcular o f1-score manualmente
f1_score = 2 * (precisao * recall) / (precisao + recall)
#Imprime o F1-score
print( '\nF1-score \n',f1_score)


# Calcular a precisão por classe manualmente
precisao_por_classe = np.zeros(numero_classes)
for classe in range(numero_classes):
    verdadeiros_positivos = matriz_confusao[classe][classe]
    falsos_positivos = np.sum(matriz_confusao[:, classe]) - verdadeiros_positivos
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    precisao_por_classe[classe] = precisao


# Imprimir a precisão por classe
#Imprime o F1-score
print( '\nPrecisão por Classe')
for classe, precisao in enumerate(precisao_por_classe):
    print(f"Classe {classe + 1}: Precisão por Classe = {precisao}")


# Calcular a precisão e recall por classe manualmente
precisao_por_classe = np.zeros(numero_classes)
recall_por_classe = np.zeros(numero_classes)
f_measure_por_classe = np.zeros(numero_classes)


for classe in range(numero_classes):
    verdadeiros_positivos = matriz_confusao[classe][classe]
    falsos_positivos = np.sum(matriz_confusao[:, classe]) - verdadeiros_positivos
    falsos_negativos = np.sum(matriz_confusao[classe, :]) - verdadeiros_positivos


precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)


f_measure = 2 * (precisao * recall) / (precisao + recall)


precisao_por_classe[classe] = precisao
recall_por_classe[classe] = recall
f_measure_por_classe[classe] = f_measure


# Imprimir a medida-F por classe
print( '\nPrecisão por Medida-F')
for classe, f_measure in enumerate(f_measure_por_classe):
    print(f"Classe {classe + 1}: Medida-F por Classe = {f_measure}")


# Calcular a taxa de FP por classe manualmente
taxa_fp_por_classe = np.zeros(numero_classes)


for classe in range(numero_classes):
    fp = np.sum(matriz_confusao[classe, :]) - matriz_confusao[classe, classe]
    n = np.sum(matriz_confusao[classe, :]) - matriz_confusao[classe, classe]


    fpr = fp / n


taxa_fp_por_classe[classe] = fpr


# Imprimir a taxa de FP por classe
print( '\nPrecisão por taxa de FP por classe')
for classe, taxa_fp in enumerate(taxa_fp_por_classe):
    print(f"Classe {classe + 1}: Taxa de FP por Classe = {taxa_fp}")




