import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import pdfplumber
from io import BytesIO

def get_normativo_bacen(tipo, numero):
    tipo_norma = tipo.lower()
    
   
    if "comunic" in tipo_norma:
        endpoint = "exibeoutrasnormas"
        tipo_api = "COMUNICADO"
    elif "resolu" in tipo_norma:
        endpoint = "exibenormativo"
        tipo_api = "Resolução BCB"
    elif "instr" in tipo_norma:
        endpoint = "exibenormativo"
        tipo_api = "Instrução Normativa BCB"
    else:
        return None

    url = f"https://www.bcb.gov.br/api/conteudo/app/normativos/{endpoint}"
    params = {"p1": tipo_api, "p2": numero}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Referer": f"https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo={tipo_api}&numero={numero}",
        "Accept": "application/json, text/plain, */*"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if "conteudo" in data and len(data["conteudo"]) > 0:
            html_texto = data["conteudo"][0]["Texto"]
            soup = BeautifulSoup(html_texto, "lxml")
            texto = soup.get_text(separator=" ", strip=True)
            return texto
    except Exception as e:
        print(f"Erro API Bacen ({tipo} {numero}): {e}")
        return None
    return None


def get_pdf_text(content_bytes):
    try:
        with pdfplumber.open(BytesIO(content_bytes)) as pdf:
            texto = " ".join([page.extract_text() or "" for page in pdf.pages])

        return re.sub(r'\\s+', ' ', texto).strip()
    except Exception:

        return None


def get_html_text(content_text):
    soup = BeautifulSoup(content_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    cleaned_text = soup.get_text(separator=" ", strip=True)
    return " ".join(cleaned_text.split())

def get_from_link(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return get_pdf_text(resp.content)
        else:
            return get_html_text(resp.text)
    except Exception:
        return None

### ATUALIZANDO AQUI PARA NOVO DF COM MAIS DADOS
df_original = pd.read_excel("./Dados Excel/df_atualizado.xlsx")

col_tipo = "Tipo do Normativo"
col_numero = "Número"
col_origem = "Origem"
col_link = "Link do Normativo"
col_aplic = "Aplicável ao BNDES?"
col_relev = "1ª Avaliação de Relevância (AIC - Time de Compliance)"

ordem = {0: 0, "Baixa": 1, "Média": 2, "Alta": 3}


df_original["relevancia_num"] = df_original[col_relev].map(ordem)


df_unico = df_original.loc[
    df_original.groupby(col_link)["relevancia_num"].idxmax().dropna() # adicionando a limpeza dos indices nan
].drop(columns="relevancia_num")


processed_data = []

for _, row in df_unico.iterrows():
    origem = str(row[col_origem])
    tipo = str(row[col_tipo])
    numero = str(row[col_numero])
    link = row[col_link]
    aplicavel_str = row[col_aplic]
    relevancia = row[col_relev]

    texto_normativo = None

    if "bacen" in origem.lower() or "banco central" in origem.lower():
        texto_normativo = get_normativo_bacen(tipo, numero)

    if not texto_normativo:  
        texto_normativo = get_from_link(link)

    if texto_normativo:
        aplicavel_bndes = isinstance(aplicavel_str, str) and aplicavel_str.strip().lower() == "sim"
        processed_data.append({
        "link_normativo": link,         
        "texto_normativo": texto_normativo,
        "aplicavel_bndes": aplicavel_bndes,
        "relevancia": relevancia
    })

df_final = pd.DataFrame(processed_data)

print("Total de normativos processados:", len(df_final))

df_final.to_csv("./Dados CSV/normativos_processados.csv", index=False, encoding="utf-8")




