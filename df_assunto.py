import pandas as pd 

df = pd.read_excel('df_atualizado.xlsx')
# Linhas sem assunto não são usadas
df = df[df["Assunto do Normativo"].astype(str) != "..."]
# Limpeza (tirando espaços aleatórios)
df["Assunto do Normativo"] = (df["Assunto do Normativo"].astype(str).str.replace('_x000D_', ' ', regex=False).str.replace(r'[\r\n]', ' ', regex=True))
df["Assunto do Normativo"] = (df["Assunto do Normativo"].str.replace(' +', ' ', regex=True).str.strip())


## Preparando colunas de classificação única (Aplicabilidade e Relevância)


col_assunto = "assunto_normativo"
col_link = "link_normativo"
col_aplic = "aplicavel_bndes"
col_relev = "relevancia"

df = df.rename(columns={
    "Assunto do Normativo": col_assunto,
    "Link do Normativo": col_link,
    "Aplicável ao BNDES?": col_aplic,
    "1ª Avaliação de Relevância (AIC - Time de Compliance)": col_relev
})


ordem = {"0": 0, "Baixa": 1, "Média": 2, "Alta": 3}
df["relevancia_num"] = df[col_relev].map(ordem)

df_valores_unicos = df.loc[
    df.groupby(col_link)["relevancia_num"].idxmax().dropna()
].drop(columns="relevancia_num")


cols_interesse = [col_link, col_assunto, col_aplic, col_relev]
df_valores_unicos = df_valores_unicos[cols_interesse]
df_valores_unicos = df_valores_unicos.reset_index()




## Preparando colunas de classificação multirótulos (áreas)

# DataFrame auxiliar com assunto e área 
df_areas = df[[col_assunto, 'Área']].copy()
# Cria uma coluna binária para cada área 
y_multilabel = pd.get_dummies(df_areas['Área'], prefix='Area')
df_consolidado = pd.concat([df_areas[col_assunto], y_multilabel], axis=1)

# Agrupa por assunto e soma as colunas binárias, no final cada linha é um normativo único 
# e as colunas de área indicam com 1 todas as áreas às quais o normativo pertence
df_areas_final = df_consolidado.groupby(col_assunto).sum()
# Garantindo classificação binária 
df_areas_final = (df_areas_final > 0).astype(int)
df_areas_final = df_areas_final.reset_index()


## Combinando os DataFrames 

df_final = pd.merge(
    df_valores_unicos, 
    df_areas_final, 
    on=col_assunto, 
    how='inner'
)


## Salvando em um csv

df_final = df_final.rename(columns={'1ª Avaliação de Relevância (AIC - Time de Compliance)': 'Relevância'})
df_final.to_csv('df_assunto_limpo.csv', index=False)

