import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats



#Ler os dados
dados = pd.read_csv('Skin_NonSkin.txt', sep='\t')

print(dados)

# Vetor de valores
X = dados.iloc[:, :-1].values
# Vetor de classe
y = dados.iloc[:, -1].values

print('\nVetor de valores')
print(X)

print('\nVetor de Classe')
print(y)

# Quantidade de folders
folders = 100


# Vetor de taxas de acertos
lista_acertos = []


# Vetor de medidas-F
lista_medida_f = []


# Inicializar o objeto de validação cruzada estratificada
stratified_kfold = StratifiedKFold(n_splits=folders, shuffle=True, random_state=42)


# Realizar a validação cruzada
# Passa o X vetor de valores e y como classe
# para a função fazer a partição
for trainamento_index, teste_index in stratified_kfold.split(X, y):
    # Dividir os dados em conjunto de treinamento e teste usando os índices
    X_trainamento, X_teste = X[trainamento_index], X[teste_index]
    y_trainamento, y_teste = y[trainamento_index], y[teste_index]


    # Inicializar o classificador 1-NN (distância Euclidiana)
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')


    # Treinar o classificador com os dados de treinamento
    knn.fit(X_trainamento, y_trainamento)


    # Fazer previsões no conjunto de teste
    y_pred = knn.predict(X_teste)


    # Calcular a acurácia do fold atual
    accuracy = accuracy_score(y_teste, y_pred)
    lista_acertos.append(accuracy)


    # Calcular a medida-F do fold atual
    medida_f = f1_score(y_teste, y_pred)
    lista_medida_f.append(medida_f)


# Calcular a média e o desvio padrão das acurácias
mean_accuracy = np.mean(lista_acertos)
std_accuracy = np.std(lista_acertos)


# Calcular o máximo e o mínimo das medidas-F
max_f_measure = np.max(lista_medida_f)
min_f_measure = np.min(lista_medida_f)


# Calcular a média e o desvio padrão das medidas-F
mean_f_measure = np.mean(lista_medida_f)
std_f_measure = np.std(lista_medida_f)


# Calcular o intervalo de confiança da medida-F
# Intervalo confinça 95%
intervalo_confianca = stats.t.interval(0.95, len(lista_medida_f) - 1, loc=np.mean(lista_medida_f), scale=stats.sem(lista_medida_f))




# Imprimir os resultados
print("\nMédia da medida-F é {:.4f}".format(mean_f_measure))
print("Máximo da medida-F é {:.4f}".format(max_f_measure))
print("Mínimo da medida-F é {:.4f}".format(min_f_measure))
print("\nIntervalo de confiança (95%) é {:.4f} - {:.4f}".format(intervalo_confianca[0], intervalo_confianca[1]))


# Cria o histograma da medida-F
plt.hist(lista_medida_f, bins='auto', edgecolor='black')
plt.xlabel('Medida-F')
plt.ylabel('Frequência')
plt.title('Histograma da Medida-F')
plt.grid(True)
plt.show()
