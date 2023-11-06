import pandas as pd
import numpy as np

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