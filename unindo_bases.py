import pandas as pd
from pathlib import Path

# lendo bases com pathlib
df1 = pd.read_excel(Path.cwd() / "Base de Dados YTD2025.xlsx")
df2 = pd.read_excel(Path.cwd() / "Normas 23-08-2025 a 22-09-2025 1.xlsx")

# Excluindo colunas desnecessarias
df1.drop(['2ª Avaliação de Relevância (Áreas BNDES)'], axis=1, inplace=True)
df2.drop(['Área Realizadora da Análise', 'URL', 'Informativo', 'Resposta da Área', 'Prazo', 'Data de Criação da Análise',
       'Data de Inclusão na Biblioteca', 'Etiquetas da Análise', 'Criticidade do Plano de Ação (pelo ponto focal)', 'Status da Tarefa', 'Relevância (pelo ponto focal)'], 
       axis=1,
       inplace=True)

# Renomeando colunas
df2.rename(columns={'Aplicável a Empresa':'Aplicável ao BNDES?', 'Origem do Normativo':'Origem', 'Número do Normativo': 'Número', 'Data de Sanção do Normativo':'Emissão', 'Relevância':'1ª Avaliação de Relevância (AIC - Time de Compliance)'}, inplace=True)

intersecao = df1.loc[:, df1.columns.isin(df2.columns)]
#intersecao = df1.columns.intersection(df2.columns)

# Criando e exportando df final
df_final = pd.concat([df1, df2], axis=0, ignore_index=True)
df_final.to_excel('df_atualizado.xlsx', 'ProjetoFields_BNDES')

print(df1.columns)
print(df2.columns)
print(intersecao.columns)
print(df_final.info())