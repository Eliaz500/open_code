import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########################################################
# Matriz com mais classes
#
dados = {'a': [0, 0, 0], 'b': [59, 71, 48], 'c': [0, 0, 0]}

matriz_confusao = pd.DataFrame(dados)

# Definir os nomes para as linhas
matriz_confusao.index = ['a', 'b', 'c']

# Exibir a matriz
print("\nMatriz 3x3")
print(matriz_confusao)

# Exibir a matriz de confusão como um gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.title('Matriz de Confusão')
plt.show()

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

########################################################
# Matriz com mais classes
#
dados = {'a': [59, 5, 0], 'b': [0, 62, 0], 'c': [0, 4, 48]}

matriz_confusao = pd.DataFrame(dados)



# Definir os nomes para as linhas
matriz_confusao.index = ['a', 'b', 'c']

# Exibir a matriz
print("\nMatriz 3x3")
print(matriz_confusao)

# Exibir a matriz de confusão como um gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.title('Matriz de Confusão')
plt.show()

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
total_erro = soma_total - soma_diagonal

print(f"\nQuantidade de erros {total_erro}")

# calcular a taxa de erro
# diagonal invertida divida por todos os valores
taxa_erro = total_erro / soma_total
print(f"Taxa de erro da matriz confusão {taxa_erro}")

# Calcular o recall
recall_a = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1] + matriz_confusao[0, 2])
print(f"\nO valor do Recall (a) {matriz_confusao[0, 0]} / ({matriz_confusao[0, 0]} + {matriz_confusao[0, 1]} + {matriz_confusao[0, 2]}) é {recall_a}")

# Calcular o recall
recall_b = matriz_confusao[1, 1] / (matriz_confusao[1, 0] + matriz_confusao[1, 1] + matriz_confusao[1, 2])
print(f"O valor do Recall (b) {matriz_confusao[1, 1]} / ({matriz_confusao[1, 0]} + {matriz_confusao[1, 1]} + {matriz_confusao[1, 2]}) é {recall_b}")

# Calcular o recall
recall_c = matriz_confusao[2, 2] / (matriz_confusao[2, 2] + matriz_confusao[2, 0] + matriz_confusao[2, 1])
print(f"O valor do Recall (c) {matriz_confusao[2, 2]} / ({matriz_confusao[2, 0]} + {matriz_confusao[2, 1]} + {matriz_confusao[2, 2]}) é {recall_c}")

# Calcular o Precisão
precisao_a = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[1, 0] + matriz_confusao[2, 0])
print(f"\nO valor da Precisão (a) {matriz_confusao[0, 0]} / ({matriz_confusao[0, 0]} + {matriz_confusao[1, 0]} + {matriz_confusao[2, 0]}) é {precisao_a}")

# Calcular o Precisão
precisao_b = matriz_confusao[1, 1] / (matriz_confusao[1, 1] + matriz_confusao[0, 1] + matriz_confusao[2, 1])
print(f"O valor da Precisão (b) {matriz_confusao[1, 1]} / ({matriz_confusao[0, 1]} + {matriz_confusao[1, 1]} + {matriz_confusao[2, 1]}) é {precisao_b}")

# Calcular o Precisão
precisao_c = matriz_confusao[2, 2] / (matriz_confusao[2, 2] + matriz_confusao[0, 2] + matriz_confusao[1, 2])
print(f"O valor da Precisão (c) {matriz_confusao[2, 2]} / ({matriz_confusao[0, 2]} + {matriz_confusao[1, 2]} + {matriz_confusao[2, 2]}) é {precisao_c}")

medida_F_a = 2 * precisao_a * recall_a / (precisao_a + recall_a)
print(f"\nO valor da Medida F (a) {medida_F_a}")

medida_F_b = 2 * precisao_b * recall_b / (precisao_b + recall_b)
print(f"O valor da Medida F (b) {medida_F_b}")

medida_F_c = 2 * precisao_c * recall_c / (precisao_c + recall_c)
print(f"O valor da Medida F (c) {medida_F_c}")
