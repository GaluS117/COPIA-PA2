#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/06_analise_informes.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/06_analise_informes.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# 1. Objetivo
# ==========================================================================================
# ============================================================
# 1. OBJETIVO
# ============================================================
ENTRADA_PREVISTA = "db/bases_tratadas/informes/informes_corpus_2025.csv"
SAIDAS_PREVISTAS = ["informes_frequencia_termos.csv", "informes_mencoes.csv", "informes_tfidf_por_informe.csv", "informes_indicadores_citados.csv", "informes_amostra_revisao.csv", "informes_sintese_por_informe.csv"]

# ==========================================================================================
# 2. Bibliotecas
# ==========================================================================================
# ============================================================
# 2. BIBLIOTECAS
# ============================================================
import re, json, unicodedata, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore"); plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})
pd.set_option("display.width", 160, "display.max_colwidth", 100)
SEED = 42

# ==========================================================================================
# 3. Caminhos
# ==========================================================================================
# ============================================================
# 3. CAMINHOS
# ============================================================
RAIZ = Path(__file__).resolve().parents[1]
TRATADA = RAIZ/"db/bases_tratadas/informes"
FIG = RAIZ/"reports/figuras"; FIG.mkdir(parents=True, exist_ok=True)
ARQ_CORPUS = TRATADA/"informes_corpus_2025.csv"
ARQ_SECOES = TRATADA/"informes_secoes_detectadas.csv"
ARQ_IBGE   = RAIZ/"db/bases_tratadas/ibge/ibge_municipio_2022_tratado.csv"
SAIDA = {n: TRATADA/n for n in SAIDAS_PREVISTAS}
ARQ_LOG = TRATADA/"informes_analise_log.json"
LOG = {"entrada": str(ARQ_CORPUS.relative_to(RAIZ)), "etapas": {}}
print("corpus existe:", ARQ_CORPUS.exists(), "| IBGE tratado existe:", ARQ_IBGE.exists())

# ==========================================================================================
# 4. Carregamento
# ==========================================================================================
# ============================================================
# 4. CARREGAMENTO
# ============================================================
if ARQ_CORPUS.exists():
    corpus = pd.read_csv(ARQ_CORPUS, dtype={"texto_original": str, "texto_limpo": str, "texto_dengue": str}).fillna({"texto_limpo": "", "texto_dengue": ""})
else:
    corpus = pd.DataFrame(columns=["informe","pagina","sem_texto","n_caracteres","texto_original","texto_limpo","secao_dengue","texto_dengue"])
    print(">>> corpus não encontrado: execute 04_limpeza_informes.ipynb com os PDFs em db/bases_brutas/informes/")
paginas = []
for inf, g in corpus.groupby("informe"):
    tem_secao = (g.secao_dengue == 1).any()
    for _, r in g.iterrows():
        txt = r.texto_dengue if tem_secao else r.texto_limpo
        if tem_secao and r.secao_dengue != 1: continue
        tipo = r.tipo_secao_dengue if ("tipo_secao_dengue" in g.columns and isinstance(r.tipo_secao_dengue, str) and r.tipo_secao_dengue) else "narrativa"
        paginas.append({"informe": inf, "pagina": int(r.pagina), "fallback_texto_completo": int(not tem_secao), "tipo": tipo, "texto": txt})
paginas = pd.DataFrame(paginas, columns=["informe","pagina","fallback_texto_completo","tipo","texto"])
narr = paginas[paginas.tipo.str.contains("narrativa")] if len(paginas) else paginas
periodo = corpus.groupby("informe")[["informe_num","se_inicio","se_fim","data_informe"]].first() if "informe_num" in corpus.columns else pd.DataFrame()
if len(periodo): paginas = paginas.merge(periodo, left_on="informe", right_index=True, how="left").sort_values(["informe_num","pagina"]).reset_index(drop=True)
print(f"{corpus.informe.nunique()} informe(s), {len(corpus)} páginas no corpus, {len(paginas)} páginas de dengue: {len(narr)} narrativas/infográfico e {len(paginas)-len(narr)} de tabelas anexas ({int(paginas.fallback_texto_completo.sum()) if len(paginas) else 0} por fallback)")
LOG["etapas"]["carregamento"] = {"informes": int(corpus.informe.nunique()), "paginas_corpus": int(len(corpus)), "paginas_analisadas": int(len(paginas))}

# ==========================================================================================
# 5. Diagnóstico do corpus
# ==========================================================================================
# ============================================================
# 5. DIAGNÓSTICO
# ============================================================
if len(paginas):
    diag = paginas.groupby("informe").agg(paginas=("pagina","nunique"), palavras=("texto", lambda s: sum(len(t.split()) for t in s)), fallback=("fallback_texto_completo","max"))
    print(diag.to_string())
    print("\nInício da seção de dengue (primeira página analisada) por informe:")
    for inf, g in paginas.groupby("informe", sort=False): print(f"  {inf[:32]}: {g.iloc[0].texto[:150]}…")

# ==========================================================================================
# 6. Regras da análise
# ==========================================================================================
# ============================================================
# 6. REGRAS DA ANÁLISE
# ============================================================
STOP = set('''a à as ao aos o os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre entre até desde e ou mas que se
não sim mais menos muito muitos muita muitas pouco poucos é são foi foram ser será serão está estão esteve estavam estar há havia ter tem têm tinha
como quando onde qual quais quem cujo cuja este esta estes estas esse essa esses essas aquele aquela isso isto aquilo seu sua seus suas nosso nossa
ele ela eles elas nós eu tu você vocês lhe lhes me te nos já ainda também só apenas mesmo mesma mesmos mesmas outro outra outros outras todo toda todos
todas cada ambos ambas pelo pela pelos pelas num numa nele nela neste nesta nesse nessa deste desta desse dessa dele dela ante após contra durante
perante segundo conforme além dentro fora acima abaixo assim então portanto porém contudo entretanto todavia logo pois porque porquê ano anos dia dias
mês meses vs figura tabela fonte gráfico quadro pág pagina página nº n º'''.split())
ESTRUTURA = {"figura","tabela","fonte","grafico","quadro","boletim","informe","ase","sinan","sinanon","line"}

RE_FONTE = re.compile(r"Fonte:[^.]*\.(?:\s*Dados sujeitos a altera[cç][aã]o\.)?", re.I)      # frases de fonte dos gráficos
RE_GLUED = re.compile(r"\b\d\S{25,}")                                       # notas de rodapé com palavras coladas (ex.: 1Dadosatualizadosem…)
def sem_fonte(t): return RE_GLUED.sub(" ", RE_FONTE.sub(" ", str(t)))
def normalizar(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z\s]", " ", t)
STOP = {normalizar(w).strip() for w in STOP}      # mesma normalização dos tokens (não -> nao, será -> sera)
def tokens(t):
    return [w for w in normalizar(sem_fonte(t)).split() if len(w) > 2 and w not in STOP and w not in ESTRUTURA]

EXPRESSOES = {
 "sorotipo":   r"\b(denv[\s\-]?[1234]|sorotipos?)\b",
 "tendencia":  r"\b(aumento|aumentou|elevacao|reducao|reduziu|queda|caiu|estabilidade|estavel|crescimento|cresceu|declinio|tendencia)\b",
 "gravidade":  r"\b(obitos?|graves?|alarme|internac(ao|oes)|letalidade)\b",
 "vigilancia": r"\b(incidencia|notificac(ao|oes)|notificados?|confirmados?|provaveis|prov[aá]vel|semanas? epidemiologicas?)\b",
}
CONTEXTO = 70   # caracteres de cada lado do trecho

def mencoes_por_regex(df, nome, padrao):
    out = []
    for _, r in df.iterrows():
        norm = normalizar(r.texto)
        for m in re.finditer(padrao, norm):
            out.append({"informe": r.informe, "pagina": r.pagina, "grupo": nome, "termo": m.group(0),
                        "trecho": r.texto[max(0, m.start()-CONTEXTO): m.end()+CONTEXTO].replace("\n", " ")})
    return out
# R3 — localidades a partir do IBGE tratado
if ARQ_IBGE.exists():
    ibge = pd.read_csv(ARQ_IBGE, dtype=str)
    locais = {}
    for uf in ibge.uf.unique(): locais.setdefault(normalizar(uf).strip(), []).append(("uf", uf, None))          # UF e região têm prioridade:
    for rg in ibge.regiao.unique(): locais.setdefault(normalizar(rg).strip(), []).append(("regiao", rg, None))  # nos informes federais o nível é estadual
    for _, r in ibge.iterrows():
        n = normalizar(r.nome_municipio).strip()
        if len(n) >= 5: locais.setdefault(n, []).append(("municipio", f"{r.nome_municipio} ({r.uf})", r.cod_municipio))
    print("nomes de localidade no dicionário:", len(locais), "(municípios ≥ 5 letras, UFs e regiões)")
else:
    locais = {}; print("IBGE tratado ausente: execute 03_limpeza_ibge.ipynb para identificar localidades")
padrao_locais = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in locais), key=len, reverse=True)) + r")\b") if locais else None

# ==========================================================================================
# 7. Aplicação
# ==========================================================================================
# ============================================================
# 7. APLICAÇÃO
# ============================================================
freq_geral, freq_inf, mencoes = Counter(), {}, []
for inf, g in narr.groupby("informe"):
    c = Counter(w for t in g.texto for w in tokens(t)); freq_inf[inf] = c; freq_geral.update(c)
for nome, padrao in EXPRESSOES.items(): mencoes += [dict(m, tipo_pagina="narrativa") for m in mencoes_por_regex(narr, nome, padrao)]
if padrao_locais is not None:
    for _, r in paginas.iterrows():
        norm = normalizar(sem_fonte(r.texto))
        for m in padrao_locais.finditer(norm):
            tipo, nome, cod = locais[m.group(0)][0]
            mencoes.append({"informe": r.informe, "pagina": r.pagina, "grupo": f"localidade_{tipo}", "termo": nome, "tipo_pagina": r.tipo,
                            "trecho": r.texto[max(0, m.start()-CONTEXTO): m.end()+CONTEXTO].replace("\n", " "), "cod_municipio": cod,
                            "homonimos": len(locais[m.group(0)])})
mencoes = pd.DataFrame(mencoes, columns=["informe","pagina","grupo","termo","tipo_pagina","trecho","cod_municipio","homonimos"])
freq_tab = pd.DataFrame(freq_geral.most_common(40), columns=["termo","frequencia"])
print("termos mais frequentes na narrativa de dengue (após stopwords):"); print(freq_tab.head(25).to_string(index=False))
print("\nmenções por grupo e tipo de página:"); print(pd.crosstab(mencoes.grupo, mencoes.tipo_pagina).to_string() if len(mencoes) else "nenhuma")
print("\nUFs mais citadas na narrativa:"); print(mencoes[(mencoes.grupo == "localidade_uf") & (mencoes.tipo_pagina == "narrativa")].termo.value_counts().head(12).to_string() if len(mencoes) else "")
# R4 — TF-IDF opcional, um documento por informe
docs = narr.groupby("informe").texto.apply(lambda s: " ".join(normalizar(sem_fonte(t)) for t in s))
tfidf_top = pd.DataFrame(columns=["informe","termo","peso"])
if len(docs) >= 2:
    vec = TfidfVectorizer(stop_words=sorted(STOP | ESTRUTURA), ngram_range=(1, 2), min_df=2, max_df=0.8, token_pattern=r"(?u)\b[a-z]{3,}\b")   # max_df exclui o que aparece em >80% dos informes
    X = vec.fit_transform(docs); termos = vec.get_feature_names_out()
    linhas = []
    for i, inf in enumerate(docs.index):
        row = X[i].toarray().ravel(); top = row.argsort()[::-1][:10]
        linhas += [{"informe": inf, "termo": termos[j], "peso": round(float(row[j]), 4)} for j in top if row[j] > 0]
    tfidf_top = pd.DataFrame(linhas)
    print("termos mais característicos por informe (TF-IDF):"); print(tfidf_top.groupby("informe").termo.apply(lambda s: ", ".join(s)).to_string())
else:
    print("TF-IDF requer ao menos 2 informes")
# R6 — indicadores citados na narrativa de cada informe (acumulados SE 1 -> SE final; não somar entre informes)
RE_IND = {
 "casos_provaveis":      r"foram notificados\s+([\d\.]+)\s+casos prov[aá]veis de\s+dengue",
 "incidencia_100mil":    r"coeficiente de incid[eê]ncia de\s+([\d\.,]+)\s+casos\s*/\s*100",
 "variacao_vs_2024_pct": r"(redu[cç][aã]o|aumento) de\s+([\d\.,]+)\s*%\s+no n[uú]mero de casos prov[aá]veis",
 "obitos_confirmados":   r"foram confirmados\s+([\d\.]+)\s+[oó]bitos",
 "obitos_investigacao":  r"([\d\.]+)\s+est[aã]o em investiga[cç][aã]o",
 "tendencia_nowcasting": r"(tend[eê]ncia (?:de |à )(?:alta|queda|estabilidade|redu[cç][aã]o)[^.]*|estimativa (?:por )?nowcasting[^.]*)",
}
def num_br(x): return float(str(x).replace(".", "").replace(",", "."))
indic = []
for inf, g in narr.groupby("informe", sort=False):
    texto = " ".join(g.texto); linha = {"informe": inf}
    for k, pat in RE_IND.items():
        m = re.search(pat, texto, re.I)
        if not m: linha[k] = np.nan; continue
        if k == "variacao_vs_2024_pct": linha[k] = (-1 if m.group(1).lower().startswith("redu") else 1) * num_br(m.group(2))
        elif k == "tendencia_nowcasting": linha[k] = m.group(1)[:110]
        else: linha[k] = num_br(m.group(1))
    soro = sorted(set(x.upper().replace(" ", "").replace("DENV", "DENV-").replace("--", "-") for x in re.findall(r"denv[\s\-]?[1234]", texto, re.I)))
    linha["sorotipos_citados"] = ", ".join(soro)
    pred = re.search(r"predomin[aâ]ncia d[oe]\s+(denv[\s\-]?\d)", texto, re.I); linha["sorotipo_predominante"] = pred.group(1).upper().replace(" ", "-") if pred else ""
    indic.append(linha)
indic = pd.DataFrame(indic)
if len(periodo) and len(indic): indic = indic.merge(periodo, left_on="informe", right_index=True, how="left").sort_values("informe_num")
indic.drop(columns=["informe"], errors="ignore")
if len(mencoes):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].barh(freq_tab.termo.head(20)[::-1], freq_tab.frequencia.head(20)[::-1], color="steelblue"); ax[0].set_title("Termos mais frequentes na narrativa de dengue")
    g = mencoes[mencoes.grupo.isin(["sorotipo","tendencia","gravidade"]) & (mencoes.tipo_pagina == "narrativa")].groupby(["informe","grupo"]).size().unstack(fill_value=0)
    if len(periodo): g.index = g.index.map(lambda i: f"nº {int(periodo.loc[i, 'informe_num']):02d}"); g = g.sort_index()
    g.plot(kind="bar", ax=ax[1], width=.8); ax[1].set_title("Menções por informe: sorotipo, tendência e gravidade"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    plt.tight_layout(); plt.savefig(FIG/"18_informes_termos_mencoes.png", dpi=120); plt.show()
if len(indic) and indic.casos_provaveis.notna().any():
    fig, ax1 = plt.subplots(figsize=(11, 4.2)); x = indic.informe_num.astype(int)
    ax1.plot(x, indic.casos_provaveis/1e3, "o-", color="steelblue", label="casos prováveis acumulados (mil)"); ax1.set_ylabel("casos prováveis (mil)", color="steelblue"); ax1.set_xlabel("informe (nº) — cada ponto é o acumulado SE 1 até a SE final daquele informe")
    ax2 = ax1.twinx(); ax2.plot(x, indic.obitos_confirmados, "s--", color="crimson", label="óbitos confirmados"); ax2.set_ylabel("óbitos confirmados", color="crimson")
    ax1.set_xticks(x); ax1.set_title("19. Trajetória de 2025 relatada pelos informes do COE (valores citados no texto, não somados)")
    plt.tight_layout(); plt.savefig(FIG/"19_informes_trajetoria_citada.png", dpi=120); plt.show()

# ==========================================================================================
# 8. Revisão manual (R5)
# ==========================================================================================
# ============================================================
# 8. REVISÃO MANUAL
# ============================================================
amostra = mencoes.sample(min(30, len(mencoes)), random_state=SEED).sort_values(["informe","pagina"]) if len(mencoes) else mencoes.copy()
amostra["revisao"] = ""; amostra["observacao"] = ""
amostra[["informe","pagina","grupo","termo","trecho","revisao","observacao"]].head(15)

# ==========================================================================================
# 9. Exportação
# ==========================================================================================
# ============================================================
# 9. EXPORTAÇÃO
# ============================================================
if len(paginas):
    pd.DataFrame([{"informe": inf, "termo": t, "frequencia": n} for inf, c in freq_inf.items() for t, n in c.most_common(50)]).to_csv(SAIDA["informes_frequencia_termos.csv"], index=False)
    mencoes.to_csv(SAIDA["informes_mencoes.csv"], index=False)
    tfidf_top.to_csv(SAIDA["informes_tfidf_por_informe.csv"], index=False)
    amostra.to_csv(SAIDA["informes_amostra_revisao.csv"], index=False)
    sintese = paginas.groupby("informe").agg(paginas_analisadas=("pagina","nunique"), fallback=("fallback_texto_completo","max"), palavras=("texto", lambda s: sum(len(t.split()) for t in s))).reset_index()
    if len(periodo): sintese = sintese.merge(periodo, left_on="informe", right_index=True, how="left").sort_values("informe_num")
    indic.to_csv(SAIDA["informes_indicadores_citados.csv"], index=False)
    for grp in ["sorotipo","tendencia","gravidade","vigilancia","localidade_uf","localidade_regiao","localidade_municipio"]:
        sub = mencoes[(mencoes.grupo == grp) & (mencoes.tipo_pagina == "narrativa")]
        sintese[grp] = sintese.informe.map(sub.groupby("informe").size()).fillna(0).astype(int)
        sintese[grp + "_termos"] = sintese.informe.map(sub.groupby("informe").termo.apply(lambda s: ", ".join(pd.Series(s).value_counts().head(5).index))).fillna("")
    sintese["top_termos"] = sintese.informe.map({inf: ", ".join(t for t, _ in c.most_common(8)) for inf, c in freq_inf.items()})
    sintese.to_csv(SAIDA["informes_sintese_por_informe.csv"], index=False)
    LOG["etapas"]["exportacao"] = {k: str(v.relative_to(RAIZ)) for k, v in SAIDA.items()}; print("saídas gravadas em", TRATADA.relative_to(RAIZ))
    display(sintese)
else:
    LOG["etapas"]["exportacao"] = "não executada: corpus vazio"; print("nada gravado")
LOG["executado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
ARQ_LOG.write_text(json.dumps(LOG, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# ==========================================================================================
# 10. Resumo
# ==========================================================================================
# ============================================================
# 10. RESUMO
# ============================================================
pd.DataFrame([
    ["informes analisados", int(paginas.informe.nunique()) if len(paginas) else 0, ""],
    ["páginas analisadas (seção de dengue)", len(paginas), f"{len(narr)} narrativa, {len(paginas)-len(narr)} tabela anexa"],
    ["termos distintos (após stopwords)", len(freq_geral), ""],
    ["menções registradas", len(mencoes), "com informe, página e trecho"],
    ["menções de localidade na narrativa", int((mencoes.grupo.str.startswith("localidade") & (mencoes.tipo_pagina == "narrativa")).sum()) if len(mencoes) else 0, "candidatas; revisar homônimos"],
    ["menções de localidade em tabelas anexas", int((mencoes.grupo.str.startswith("localidade") & (mencoes.tipo_pagina != "narrativa")).sum()) if len(mencoes) else 0, "listas de UF, não interpretadas"],
    ["informes com indicadores citados extraídos", int(indic.casos_provaveis.notna().sum()) if len(indic) else 0, "R6"],
    ["amostra para revisão manual", len(amostra), "coluna revisao a preencher"],
], columns=["item","valor","observação"])
