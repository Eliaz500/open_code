import pandas as pd
import matplotlib.pyplot as plt

nome_colunas = ['sepal lenght', 'sepal width', 'petal lenght', 'petal width', 'class']

dados = pd.read_csv('iris.data', header=None, names= nome_colunas)

print(dados)

sepal_width = dados['sepal width']
petal_lenght = dados['petal lenght']

print(sepal_width)
print(petal_lenght)

classe = dados['class']

print(classe)

# Tamanho da imagem do gráfico
plt.figure(figsize=(8,6))

# Cria o gráfico
plt.scatter(sepal_width[classe == 'Iris-setosa'], petal_lenght[classe == 'Iris-setosa'], c='red',
            marker='o', s=80, label= 'Iris-setosa')
plt.scatter(sepal_width[classe == 'Iris-versicolor'], petal_lenght[classe == 'Iris-versicolor'], c='green',
            marker='o', s=80, label= 'Iris-versicolor')
plt.scatter(sepal_width[classe == 'Iris-virginica'], petal_lenght[classe == 'Iris-virginica'], c='blue',
            marker='o', s=80, label= 'Iris-virginica')

# Nomeando a Legenda
plt.xlabel('Sepal Width')
plt.ylabel('Petal Lenght')
plt.title('Iris dados: Sepal Width x Petal Lenght')

# Adicionando a legenda
plt.legend()

# Abre o gráfico
plt.show()

