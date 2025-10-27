import pandas as pd 

df = pd.DataFrame(pd.read_excel('df_atualizado.xlsx'))
# Linhas sem assunto não são usadas
df = df[df["Assunto do Normativo"].astype(str) != "..."]


## Preparando colunas de classificação única (Aplicabilidade e Relevância)

cols_interesse = ["Aplicável ao BNDES?", "1ª Avaliação de Relevância (AIC - Time de Compliance)"]
# Agrupa por assunto e pega a primeira linha das colunas de interesse de cada grupo 
df_valores_unicos = df.groupby('Assunto do Normativo')[cols_interesse].first()
df_valores_unicos = df_valores_unicos.reset_index()


## Preparando colunas de classificação multirótulos (áreas)

# DataFrame auxiliar com assunto e área 
df_areas = df[['Assunto do Normativo', 'Área']].copy()
# Cria uma coluna binária para cada área 
y_multilabel = pd.get_dummies(df_areas['Área'], prefix='Area')
df_consolidado = pd.concat([df_areas['Assunto do Normativo'], y_multilabel], axis=1)

# Agrupa por assunto e soma as colunas binárias, no final cada linha é um normativo único 
# e as colunas de área indicam com 1 todas as áreas às quais o normativo pertence
df_areas_final = df_consolidado.groupby('Assunto do Normativo').sum()
# Garantindo classificação binária 
df_areas_final = (df_areas_final > 0).astype(int)
df_areas_final = df_areas_final.reset_index()


## Combinando os DataFrames 

df_final = pd.merge(
    df_valores_unicos, 
    df_areas_final, 
    on='Assunto do Normativo', 
    how='inner'
)


## Salvando em um csv

df_final = df_final.rename(columns={'1ª Avaliação de Relevância (AIC - Time de Compliance)': 'Relevância'})
df_final.to_csv('df_assunto_limpo.csv', index=False)

