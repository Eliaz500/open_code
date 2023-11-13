import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score
from scipy import stats

nome_coluna = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash', ' Magnesium',
' Total phenols', 'Flavanoids', 'Nonflavanoid phenols','Proanthocyanins',
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#Ler os dados
dados_wine = pd.read_csv('wine.data', header=None, names= nome_coluna)


# Separar as features (X) e os rótulos (y)
X = dados_wine.iloc[:, 1:]
y = dados_wine['class']

# Inicializar os classificadores
knn_1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_3_peso = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')

# Número de repetições
n_repeticoes = 100

# Listas para armazenar as diferenças nas taxas de acerto
diferencas_taxas_acerto = []

# Listas para armazenar as taxas de acerto de cada classificador
taxas_acerto_1 = []
taxas_acerto_3_peso = []

# Loop de repetições
for _ in range(n_repeticoes):
    # Dividir os dados em conjunto de treinamento e teste (holdout)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Treinar e fazer previsões com 1-NN
    knn_1.fit(X_train, y_train)
    y_pred_1 = knn_1.predict(X_test)
    taxa_acerto_1 = accuracy_score(y_test, y_pred_1)
    taxas_acerto_1.append(taxa_acerto_1)

    # Treinar e fazer previsões com 3-NN com peso
    knn_3_peso.fit(X_train, y_train)
    y_pred_3_peso = knn_3_peso.predict(X_test)
    # Verificar se há alguma previsão correta antes de calcular a taxa de acerto
    if any(y_test == y_pred_3_peso):
        taxa_acerto_3_peso = accuracy_score(y_test, y_pred_3_peso)
        taxas_acerto_3_peso.append(taxa_acerto_3_peso)
    else:
        print("Aviso: Nenhuma previsão correta para 3-NN com peso nesta iteração.")

    # Calcular a diferença nas taxas de acerto
    diferenca_taxa = taxa_acerto_1 - taxa_acerto_3_peso
    diferencas_taxas_acerto.append(diferenca_taxa)

# Remover valores não numéricos das diferenças
diferencas_taxas_acerto = [diferenca for diferenca in diferencas_taxas_acerto if not np.isnan(diferenca)]
# Arredondar as diferenças para evitar problemas numéricos
diferencas_taxas_acerto_arredondadas = np.round(diferencas_taxas_acerto, 5)



# Calcular a média das diferenças nas taxas de acerto
media_diferencas = np.mean(diferencas_taxas_acerto)

# Imprimir a média das diferenças
print("\nMédia das diferenças nas taxas de acerto:", media_diferencas)

# Calcular o intervalo de confiança das diferenças nas taxas de acerto
intervalo_confianca = stats.t.interval(0.95, len(diferencas_taxas_acerto_arredondadas) - 1,
                                       loc=np.mean(diferencas_taxas_acerto_arredondadas),
                                       scale=stats.sem(diferencas_taxas_acerto_arredondadas))

# Imprimir o intervalo de confiança
print("\nIntervalo de Confiança das Diferenças:", intervalo_confianca)


# Teste de hipótese
estatistica, valor_p = stats.ttest_1samp(diferencas_taxas_acerto, popmean=0)

# Nível de significância
significancia = 0.05

# Conclusão do teste
if valor_p < significancia:
    print('\nRejeita H0: As diferenças entre 1-NN e 3-NN com peso são significativas.')
else:
    print('\nNão Rejeita H0: Não há evidências suficientes para concluir que as diferenças são significativas.')

# Calcular o intervalo de confiança da taxa de acerto do 1-NN
intervalo_confianca_1 = stats.t.interval(0.95, len(taxas_acerto_1) - 1,
                                       loc=np.mean(taxas_acerto_1),
                                       scale=stats.sem(taxas_acerto_1))

# Imprimir os intervalos de confiança
print("\nIntervalo de Confiança da Taxa de Acerto do 1-NN:", intervalo_confianca_1)

# Calcular o intervalo de confiança da taxa de acerto do 3-NN com peso
if len(taxas_acerto_3_peso) > 1:  # Verificar se há mais de uma amostra para calcular o intervalo de confiança
    intervalo_confianca_3_peso = stats.t.interval(0.95, len(taxas_acerto_3_peso) - 1,
                                                  loc=np.mean(taxas_acerto_3_peso),
                                                  scale=stats.sem(taxas_acerto_3_peso))
    # Imprimir o intervalo de confiança
    print("Intervalo de Confiança da Taxa de Acierto do 3-NN com Peso:", intervalo_confianca_3_peso)
else:
    print("Não há dados suficientes para calcular o intervalo de confiança do 3-NN com Peso.")

# Teste de hipótese para sobreposição dos intervalos de confiança
alpha_corrigido = 0.05  # Nível de significância ajustado

# Comparar intervalo de confiança do 1-NN com 3-NN com peso
_, p_valor_intervalos = stats.ttest_ind(taxas_acerto_1, taxas_acerto_3_peso, equal_var=False)

# Aplicar correção de Bonferroni ao valor p
rejeitar_hipotese_nula, p_valor_corrigido, _, _ = multipletests([p_valor_intervalos], alpha=alpha_corrigido, method='b')

# Conclusão do teste de sobreposição de intervalos de confiança
if rejeitar_hipotese_nula:
    print('\nRejeita H0: Existe diferença significativa entre as taxas de acerto do 1-NN e 3-NN com peso.')
else:
    print('\nNão Rejeita H0: Não há evidências suficientes para concluir que há diferença significativa.')



