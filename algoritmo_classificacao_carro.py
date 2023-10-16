import pandas as pd


tabela = pd.read_csv('car.data', header=6)

#Nomes das colunas
tabela.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

print(tabela)

quantidade_unacc = 0
quantidade_acc = 0
quantidade_good = 0
quantidade_vgood = 0


for linha in tabela.itertuples():
    #print(linha)
    if linha.buying == 'vhigh' and linha.maint == 'vhigh' or linha.maint == 'high':
        quantidade_unacc += 1
    elif linha.buying == 'vhigh' and linha.maint == 'med' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'vhigh' and linha.maint == 'med':
        quantidade_acc += 1
    elif linha.buying == 'high' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'high' and linha.safety == 'high' or linha.safety == 'med':
        quantidade_acc += 1
    elif linha.buying == 'med' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'med' and linha.lug_boot == 'med' and linha.safety == 'high' or linha.safety == 'med':
        quantidade_acc += 1
    elif linha.buying == 'med' and linha.maint == 'med' and linha.safety == 'high' or linha.safety == 'med':
        quantidade_acc += 1
    elif linha.buying == 'med' and linha.lug_boot == 'big' and linha.safety == 'high':
        quantidade_vgood += 1
    elif linha.buying == 'med' and linha.maint == 'low' and linha.lug_boot == 'big' or linha.lug_boot == 'small' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'med' and linha.maint == 'low' and linha.lug_boot == 'big' and linha.safety == 'med':
        quantidade_good += 1
    elif linha.buying == 'med' and linha.maint == 'vhigh' and linha.lug_boot == 'small' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'low' and linha.maint == 'vhigh' and linha.lug_boot == 'big' or linha.lug_boot == 'med' and linha.safety == 'high' or linha.safety == 'med':
        quantidade_acc += 1
    elif linha.buying == 'low' and linha.maint == 'high' and linha.lug_boot == 'small' and linha.safety == 'low':
        quantidade_unacc += 1
    elif linha.buying == 'low' and linha.maint == 'high' and linha.lug_boot == 'big' and linha.lug_boot == 'big' or linha.lug_boot == 'med' and linha.safety == 'high' or linha.safety == 'med':
        quantidade_acc += 1
    elif linha.buying == 'low' and linha.maint == 'med' and linha.lug_boot == 'small' and linha.safety == 'med' or linha.safety == 'high':
        quantidade_unacc += 1
    elif linha.buying == 'low' and linha.maint == 'med' and linha.lug_boot == 'big' or linha.lug_boot == 'med' and linha.safety == 'big' or linha.safety == 'med':
        quantidade_good += 1
    elif linha.buying == 'low' and linha.maint == 'med' and linha.lug_boot == 'big' and linha.safety == 'high':
        quantidade_vgood += 1

print("Resultado")
print("Quantidade unacc: " + str(quantidade_unacc))
print("Quantidade acc: " + str(quantidade_acc))
print("Quantidade vgood: " + str(quantidade_vgood))
print("Quantidade good: " + str(quantidade_good))