import pandas as pd


tabela = pd.read_csv('car.data', header=6)

#Nomes das colunas
tabela.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

print(tabela)

for linha in tabela.itertuples():
    print(linha)
    for valor in linha[1:-1]:
        print(valor)
