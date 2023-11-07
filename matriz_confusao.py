import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Criar um matriz 2x2 com os nome das colunas
dados = {'P': ["VP", "FN"], 'N': ["FP", "VN"]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['P', 'N']

# Exibir a matriz
print(matriz_confusao)

#taxa_acerto = (VP + VN) / (VP + FP + FN + VN) ou ACURACIA
#taxa_acerto = (Diagonal) / (todos os valores) ou ACURACIA

#taxa_erro = (diagonal invertida) / (VP + FP + FN + VN)

print('\n')

# Fazendo o teste
# Criar um matriz com valores
dados = {'P': [23, 11], 'N': [15, 22]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['P', 'N']

# Exibir a matriz
print(matriz_confusao)

# Converter o Dados em um array NumPy
matriz_confusao = matriz_confusao.to_numpy()

# Calcular a soma da diagonal
soma_diagonal = np.trace(matriz_confusao)

print(f"\nSoma da diagonal {soma_diagonal}")

# Calcular a soma de todos os valores da matriz
soma_total = np.sum(matriz_confusao)

print(f"Soma de todos os valores {soma_total}")

# calcular a taxa de acerto
# diagonal divida por todos os valores
taxa_acerto = soma_diagonal / soma_total
print(f"Taxa de acerto da matriz confusão {taxa_acerto}")

# Calcular a soma da diagonal invertida
soma_diagonal_invertida = np.trace(np.fliplr(matriz_confusao))

print(f"\nSoma da diagonal invertida {soma_diagonal_invertida}")

# calcular a taxa de erro
# diagonal invertida divida por todos os valores
taxa_erro = soma_diagonal_invertida / soma_total
print(f"Taxa de erro da matriz confusão {taxa_erro}")

# Fazendo o teste com uma taxa de acerto 100%
# Criar um matriz com valores
dados = {'P': [25, 0], 'N': [0, 15]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['P', 'N']

# Exibir a matriz
print(matriz_confusao)

# Converter o Dados em um array NumPy
matriz_confusao = matriz_confusao.to_numpy()

# Calcular a soma da diagonal
soma_diagonal = np.trace(matriz_confusao)

print(f"\nSoma da diagonal {soma_diagonal}")

# Calcular a soma de todos os valores da matriz
soma_total = np.sum(matriz_confusao)

print(f"Soma de todos os valores {soma_total}")

# calcular a taxa de acerto
# diagonal divida por todos os valores
taxa_acerto = soma_diagonal / soma_total
print(f"Taxa de acerto da matriz confusão {taxa_acerto}")

# Calcular a soma da diagonal invertida
soma_diagonal_invertida = np.trace(np.fliplr(matriz_confusao))

print(f"\nSoma da diagonal invertida {soma_diagonal_invertida}")

# calcular a taxa de erro
# diagonal invertida divida por todos os valores
taxa_erro = soma_diagonal_invertida / soma_total
print(f"Taxa de erro da matriz confusão {taxa_erro}")

# Fazendo o teste com uma taxa de erro 100%
# Criar um matriz com valores
dados = {'P': [0, 15], 'N': [21, 0]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['P', 'N']

# Exibir a matriz
print(matriz_confusao)

# Converter o Dados em um array NumPy
matriz_confusao = matriz_confusao.to_numpy()

# Calcular a soma da diagonal
soma_diagonal = np.trace(matriz_confusao)

print(f"\nSoma da diagonal {soma_diagonal}")

# Calcular a soma de todos os valores da matriz
soma_total = np.sum(matriz_confusao)

print(f"Soma de todos os valores {soma_total}")

# calcular a taxa de acerto
# diagonal divida por todos os valores
taxa_acerto = soma_diagonal / soma_total
print(f"Taxa de acerto da matriz confusão {taxa_acerto}")

# Calcular a soma da diagonal invertida
soma_diagonal_invertida = np.trace(np.fliplr(matriz_confusao))

print(f"\nSoma da diagonal invertida {soma_diagonal_invertida}")

# calcular a taxa de erro
# diagonal invertida divida por todos os valores
taxa_erro = soma_diagonal_invertida / soma_total
print(f"Taxa de erro da matriz confusão {taxa_erro}")

########################################################
# Recall e Precisão
# Criar um matriz com valores
dados = {'P': [62, 0], 'N': [5, 59]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['P', 'N']

# Exibir a matriz
print("\nRecall e Precisão")
print(matriz_confusao)

# Exibir a matriz de confusão como um gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.title('Matriz de Confusão')
plt.show()

#METRICAS PARA A CLASSE POSITIVA
# RECALL E PRECISÃO
# Recall = VP / (VP + FN) ou VP / (Linha P)
# Precisão = VP / (VP + FP) ou VP / (coluna P)

# Calcular o recall
recall = matriz_confusao.iloc[0, 0] / (matriz_confusao.iloc[0, 0] + matriz_confusao.iloc[0, 1])

print(f"\nO valor do Recall {recall}")

# Calcular o Precisão
precisao = matriz_confusao.iloc[0, 0] / (matriz_confusao.iloc[0, 0] + matriz_confusao.iloc[1, 0])

print(f"\nO valor da Precisão {precisao}")

#################################################################################
#    Medida-F
##############################################################################
# F1  = 2 x Precisão x Recall / (Precisão + Recall)

medida_F = 2 * precisao * recall / (precisao + recall)

print(f"\nO valor da Medida F {medida_F}")



