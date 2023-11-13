
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

nome_coluna = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', ' Magnesium',
               ' Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
               'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Ler os dados
dados_wine = pd.read_csv('wine.data', header=None, names=nome_coluna)

# Separar as features (X) e os rótulos (y)
X = dados_wine.iloc[:, 1:]
y = dados_wine['class']

# Inicializar os classificadores
n_repeticoes = 1
k_values = list(range(1, 16))

# Lista para armazenar as taxas de acerto para cada valor de k
taxas_acerto_por_k = []

# Loop sobre diferentes valores de k
for k in k_values:
    # Lista para armazenar as taxas de acerto para repetições
    taxas_acerto_repeticoes = []

    # Loop de repetições
    for _ in range(n_repeticoes):
        # Dividir os dados em conjunto de treinamento e teste (holdout)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Treinar e fazer previsões com k-NN
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        taxa_acerto = accuracy_score(y_test, y_pred)
        taxas_acerto_repeticoes.append(taxa_acerto)

    # Adicionar a lista de taxas de acerto para o valor de k atual
    taxas_acerto_por_k.append(taxas_acerto_repeticoes)

# Teste estatístico de Friedman
estatistica, p_valor = stats.friedmanchisquare(*taxas_acerto_por_k)

# Nível de significância
significancia = 0.05

for i in range(len(taxas_acerto_por_k)):
    print(f'K = {i + 1} é {taxas_acerto_por_k[i]}')


# Conclusão do teste
if p_valor < significancia:
    print('Rejeita H0: Pelo menos uma das taxas de acerto é significativamente diferente.')
else:
    print('Não Rejeita H0: Não há evidências suficientes para concluir que as taxas de acerto são significativamente diferentes.')
