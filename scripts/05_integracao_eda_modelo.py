#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/05_integracao_eda_modelo.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/05_integracao_eda_modelo.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# ETAPA 1: Bibliotecas, ambiente e reprodutibilidade
# ==========================================================================================
import sys, platform, json, hashlib, gc, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "figure.dpi": 100})

def barras(ax, rotulos, valores, titulo, xlabel="", ylabel="", cor="steelblue", linha=None):
    ax.bar([str(r) for r in rotulos], valores, color=cor)
    if linha is not None: ax.axhline(linha, ls="--", c="gray", label="média geral"); ax.legend()
    ax.set_title(titulo); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
pd.set_option("display.width", 160, "display.max_columns", 60, "display.max_rows", 120)

SEED = 42
np.random.seed(SEED)

RAIZ = Path(__file__).resolve().parents[1]
TRATADA = RAIZ/"db/bases_tratadas"
PROC, FIG = TRATADA/"analitica", RAIZ/"reports/figuras"
for p in (PROC, FIG): p.mkdir(parents=True, exist_ok=True)

print("Python", platform.python_version())
for m in (np, pd, matplotlib, sklearn): print(f"{m.__name__:12s} {m.__version__}")

# ==========================================================================================
# ETAPA 2: Integridade das entradas tratadas
# ==========================================================================================
def sha256(caminho, bloco=1 << 24):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for parte in iter(lambda: f.read(bloco), b""): h.update(parte)
    return h.hexdigest()

ENTRADAS = {"SINAN tratado": TRATADA/"sinan/sinan_dengue_2025_tratado.parquet",
            "SINAN auditoria": TRATADA/"sinan/sinan_dengue_2025_auditoria.csv",
            "CNES tratado": TRATADA/"cnes/cnes_municipio_2025_tratado.csv",
            "IBGE tratado": TRATADA/"ibge/ibge_municipio_2022_tratado.csv"}
registro = pd.DataFrame([{"fonte": k, "arquivo": str(p.relative_to(RAIZ)), "tamanho_MB": round(p.stat().st_size/1e6, 1), "sha256": sha256(p)} for k, p in ENTRADAS.items()])
registro.to_csv(PROC/"integridade_entradas_tratadas.csv", index=False)
for k, p in ENTRADAS.items(): assert p.exists(), f"execute o notebook de limpeza que gera {p.name}"
print(registro)

# ==========================================================================================
# ETAPA 3: Leitura da camada tratada (P02 concluído nos notebooks de limpeza)
# ==========================================================================================
SINTOMAS = ["FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS","CONJUNTVIT","ARTRITE","ARTRALGIA","PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO"]
COMORB   = ["DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
SINT_BIN, COMORB_BIN = [c+"_bin" for c in SINTOMAS], [c+"_bin" for c in COMORB]

t0 = time.time()
sinan = pd.read_parquet(ENTRADAS["SINAN tratado"])
for c in ["sexo","gestante","raca","escolaridade","uf_res","cod_municipio"]: sinan[c] = sinan[c].astype("category")
log01 = json.loads((TRATADA/"sinan/sinan_limpeza_log.json").read_text(encoding="utf-8"))
print(f"SINAN tratado: {sinan.shape} | memória {sinan.memory_usage(deep=True).sum()/1e6:.0f} MB | {time.time()-t0:.0f} s")
print("Contagens registradas na limpeza (notebook 01):"); print(pd.Series(log01["etapas"]["alvo"]).to_string())
print(sinan.head())
sinan.info(memory_usage="deep")
cnes = pd.read_csv(ENTRADAS["CNES tratado"], dtype={"cod_municipio": str})
ibge = pd.read_csv(ENTRADAS["IBGE tratado"], dtype={"cod_municipio": str, "cd_mun_ibge7": str})
print("CNES tratado:", cnes.shape, "| chave única:", not cnes.duplicated(["cod_municipio","competencia"]).any(), "| competências:", cnes.competencia.min(), "->", cnes.competencia.max())
print("IBGE tratado:", ibge.shape, "| chave única:", ibge.cod_municipio.is_unique)
display(cnes.head(3)); ibge.head(3)

# ==========================================================================================
# ETAPA 4: Alvo e desequilíbrio (P03)
# ==========================================================================================
dist = sinan.CLASSI_FIN.value_counts().sort_index()
print(pd.DataFrame({"n": dist, "%": (dist/dist.sum()*100).round(2)}).to_string())
print(f"\ncasos elegíveis: {len(sinan):,} | positivos: {int(sinan.alvo.sum()):,} ({sinan.alvo.mean()*100:.2f}%)")
plt.figure(figsize=(7, 3.8))
barras(plt.gca(), ["10 dengue","11 c/ alarme","12 grave"], dist.values, "1. Classificação final dos casos elegíveis (escala log)", ylabel="casos")
plt.yscale("log")
plt.tight_layout(); plt.savefig(FIG/"01_classi_fin.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 5: Variáveis que revelam o alvo (validade temporal)
# ==========================================================================================
ALARME = ["ALRM_HIPOT","ALRM_PLAQ","ALRM_VOM","ALRM_SANG","ALRM_HEMAT","ALRM_ABDOM","ALRM_LETAR","ALRM_HEPAT","ALRM_LIQ"]
GRAVID = ["GRAV_PULSO","GRAV_CONV","GRAV_ENCH","GRAV_INSUF","GRAV_TAQUI","GRAV_EXTRE","GRAV_HIPOT","GRAV_HEMAT","GRAV_MELEN","GRAV_METRO","GRAV_SANG","GRAV_AST","GRAV_MIOC","GRAV_CONSC","GRAV_ORGAO"]
aud = pd.read_csv(ENTRADAS["SINAN auditoria"], usecols=["id_registro","alvo","HOSPITALIZ"] + ALARME + GRAVID, dtype="float32")
aud["alarme_qualquer"] = (aud[ALARME] == 1).any(axis=1); aud["gravidade_qualquer"] = (aud[GRAVID] == 1).any(axis=1)
print("alvo x algum sinal de alarme = 1"); print(pd.crosstab(aud.alvo, aud.alarme_qualquer, margins=True))
print("\nalvo x algum sinal de gravidade = 1"); print(pd.crosstab(aud.alvo, aud.gravidade_qualquer, margins=True))
print("\nalvo x HOSPITALIZ (1 sim, 2 não, 9 ignorado)"); print(pd.crosstab(aud.alvo, aud.HOSPITALIZ.fillna(-1).astype(int), margins=True))
del aud; gc.collect();

# ==========================================================================================
# ETAPA 6: Reserva do conjunto de teste (P03)
# ==========================================================================================
from sklearn.model_selection import train_test_split

ids_dev, ids_test = train_test_split(sinan.id_registro.values, test_size=0.20, stratify=sinan.alvo.values, random_state=SEED)
pd.DataFrame({"id_registro": np.concatenate([ids_dev, ids_test]), "conjunto": ["desenvolvimento"]*len(ids_dev) + ["teste"]*len(ids_test)}).to_csv(PROC/"split_ids.csv", index=False)
dev = sinan[sinan.id_registro.isin(ids_dev)].copy()
print(f"desenvolvimento: {len(dev):,} ({dev.alvo.mean()*100:.2f}% positivos) | teste reservado: {len(ids_test):,}")
del sinan; gc.collect();

# ==========================================================================================
# ETAPA 7: Ausências e os significados de "vazio" (P04)
# ==========================================================================================
# ausente (nulo físico) x ignorado (código 9 / categoria) x não se aplica, por variável candidata
linhas = []
for c in SINTOMAS + COMORB:   # na base tratada, nulo físico e código 9 já estão ambos como nulo (a separação está na validação do notebook 01)
    linhas.append({"variavel": c, "%_nulo_bruto": round(dev[c + "_bin"].isna().mean()*100, 3), "%_ignorado_(9)": "incluído no nulo", "%_nao_se_aplica": 0.0})
for c, orig, cod_ign, cod_na in [("sexo","CS_SEXO",None,None), ("gestante","CS_GESTANT",9,6), ("raca","CS_RACA",9,None), ("escolaridade","CS_ESCOL_N",9,10)]:
    linhas.append({"variavel": c, "%_nulo_bruto": round(dev[orig].isna().mean()*100, 3),
                   "%_ignorado_(9)": round(((dev[orig] == cod_ign).mean() if cod_ign else (dev[c] == "I").mean())*100, 2),
                   "%_nao_se_aplica": round((dev[orig] == cod_na).mean()*100, 2) if cod_na else 0.0})
for c in ["idade_anos","dias_ate_notificacao","cod_municipio","ocupacao_cbo"]:
    linhas.append({"variavel": c, "%_nulo_bruto": round(dev[c].isna().mean()*100, 3), "%_ignorado_(9)": np.nan, "%_nao_se_aplica": np.nan})
ausencias = pd.DataFrame(linhas).set_index("variavel")
ausencias.to_csv(PROC/"eda_ausencias_variaveis_candidatas.csv")
print("Ausência por variável candidata (% no desenvolvimento):"); print(ausencias.to_string())
consistencia = {
 "DT_SIN_PRI dentro de 29/12/2024 a 03/01/2026": bool(dev.DT_SIN_PRI.between("2024-12-29", "2026-01-03").all()),
 "DT_NOTIFIC >= DT_SIN_PRI (onde o intervalo é válido)": bool((dev.DT_NOTIFIC >= dev.DT_SIN_PRI)[dev.dias_ate_notificacao.notna()].all()),
 "SEM_PRI entre 202501 e 202553": bool(dev.SEM_PRI.between(202501, 202553).all()),
 "cod_municipio com 6 dígitos": bool(dev.cod_municipio.dropna().astype(str).str.len().eq(6).all()),
 "cod_municipio existe no IBGE": f"{dev.cod_municipio.isin(ibge.cod_municipio).mean()*100:.3f}% dos casos ({int((~dev.cod_municipio.isin(ibge.cod_municipio)).sum())} sem correspondência, ver Etapa 11)",
 "competencia em 2025 (após ajuste das bordas)": f"{dev.competencia.between(202501, 202512).mean()*100:.2f}% já em 2025; o restante é dez/2024 ou jan/2026 e recebe a competência mais próxima",
}
print(pd.Series(consistencia).to_string())
dev = dev.rename(columns={"dias_ate_notificacao": "atraso_notif"})
dev["y"] = dev.alvo
for c in SINTOMAS + COMORB: dev[c] = dev[c + "_bin"]          # nomes curtos para as etapas seguintes

# ==========================================================================================
# ETAPA 8: Análise das variáveis numéricas
# ==========================================================================================
print(dev[["idade_anos","atraso_notif"]].describe().round(2).T.to_string())
print("\nModa idade:", dev.idade_anos.mode()[0], "| Moda atraso:", dev.atraso_notif.mode()[0])
print("\nMediana por classe (0 = dengue, 1 = alarme/grave):")
print(dev.groupby("y")[["idade_anos","atraso_notif"]].median().round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(dev.idade_anos.dropna(), bins=60, color="steelblue"); ax[0].set_title("2. Distribuição da idade (anos)"); ax[0].set_xlabel("Idade"); ax[0].set_ylabel("Casos")
ax[1].hist(dev.atraso_notif.dropna().clip(upper=30), bins=31, color="darkorange"); ax[1].set_title("3. Atraso de notificação (dias, truncado em 30)"); ax[1].set_xlabel("Dias entre sintomas e notificação"); ax[1].set_ylabel("Casos")
plt.tight_layout(); plt.savefig(FIG/"02_03_idade_atraso_hist.png", dpi=120); plt.show()
dev["faixa_idade"] = pd.cut(dev.idade_anos, [-1, 4, 14, 29, 59, 121], labels=["0-4","5-14","15-29","30-59","60+"])
taxa_idade = dev.groupby("faixa_idade", observed=True).y.agg(casos="size", pct_positivos=lambda s: s.mean()*100).round(2)
print(taxa_idade.to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].boxplot([dev.loc[dev.y == 0, "idade_anos"].dropna(), dev.loc[dev.y == 1, "idade_anos"].dropna()], tick_labels=["0","1"], showfliers=False); ax[0].set_title("4. Idade por classe do alvo"); ax[0].set_xlabel("0 = dengue | 1 = alarme/grave"); ax[0].set_ylabel("Idade (anos)")
barras(ax[1], taxa_idade.index, taxa_idade.pct_positivos, "5. % de casos com alarme/grave por faixa etária", "Faixa etária", "% positivos", cor="indianred", linha=dev.y.mean()*100)
plt.tight_layout(); plt.savefig(FIG/"04_05_idade_alvo.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 9: Análise das variáveis categóricas
# ==========================================================================================
dev = dev.merge(ibge[["cod_municipio","regiao"]], on="cod_municipio", how="left")
dev["regiao"] = dev.regiao.fillna("ignorado").astype("category")

def tabela_categorica(df, col):
    t = df.groupby(col, observed=True).y.agg(n="size", pct_positivos=lambda s: s.mean()*100)
    t["freq_relativa_%"] = t.n / t.n.sum() * 100
    return t.round(2).sort_values("n", ascending=False)[["n","freq_relativa_%","pct_positivos"]]

for col in ["sexo","gestante","raca","escolaridade","regiao"]:
    print(f"\n== {col}"); print(tabela_categorica(dev, col).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ordem = dev.regiao.value_counts().index
barras(ax[0], ordem, dev.regiao.value_counts().loc[ordem].values, "6. Casos elegíveis por região de residência", "Região", "Casos")
t = tabela_categorica(dev, "regiao").loc[ordem]
barras(ax[1], t.index, t.pct_positivos, "7. % de casos com alarme/grave por região", "Região", "% positivos", cor="indianred", linha=dev.y.mean()*100)
plt.tight_layout(); plt.savefig(FIG/"06_07_regiao.png", dpi=120); plt.show()
t = dev.groupby("mes_sintomas").y.agg(casos="size", pct_positivos=lambda s: s.mean()*100).round(2)
fig, ax1 = plt.subplots(figsize=(11, 4.2))
ax1.bar(t.index.astype(str), t.casos, color="lightsteelblue"); ax1.set_ylabel("Casos"); ax1.set_xlabel("Mês dos primeiros sintomas (2025)")
ax2 = ax1.twinx(); ax2.plot(range(len(t)), t.pct_positivos, "o-", color="crimson"); ax2.set_ylabel("% positivos", color="crimson")
plt.title("8. Sazonalidade: casos por mês e % com alarme/grave"); plt.tight_layout(); plt.savefig(FIG/"08_sazonalidade.png", dpi=120); plt.show()
print(t.to_string())

# ==========================================================================================
# ETAPA 10: Sintomas e comorbidades em relação ao alvo
# ==========================================================================================
comp = (dev.groupby("y")[SINTOMAS + COMORB].mean()*100).T.astype("float64").round(1)
comp.columns = ["% em dengue (0)", "% em alarme/grave (1)"]
comp["diferença (p.p.)"] = (comp.iloc[:, 1] - comp.iloc[:, 0]).round(1)
comp.sort_values("diferença (p.p.)", ascending=False)
c = comp.sort_values("diferença (p.p.)")
plt.figure(figsize=(9, 7))
plt.barh(c.index, c["diferença (p.p.)"], color=np.where(c["diferença (p.p.)"] > 0, "crimson", "steelblue"))
plt.axvline(0, color="black", lw=0.8); plt.title("9. Diferença de frequência (p.p.): alarme/grave menos dengue"); plt.xlabel("pontos percentuais")
plt.tight_layout(); plt.savefig(FIG/"09_sintomas_diferenca.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 11: Integração do CNES e do IBGE tratados (P05)
# ==========================================================================================
IND_SANEAMENTO = [c for c in ibge.columns if c.startswith("pct_")]
def integrar(df):
    n0 = len(df)
    df = df.merge(ibge.drop(columns=["cd_mun_ibge7","uf","regiao"]), on="cod_municipio", how="left", validate="m:1")
    sem_ibge = df.populacao.isna()
    df["competencia_cnes"] = df.competencia.clip(lower=202501, upper=202512)
    df = df.merge(cnes.rename(columns={"competencia": "competencia_cnes"}), on=["cod_municipio","competencia_cnes"], how="left", validate="m:1")
    sem_cnes = df.n_estabelecimentos.isna()
    for c in ["n_estabelecimentos","leitos","leitos_sus","leitos_uti","leitos_uti_sus"]:
        df[c] = df[c].fillna(0).astype("float32")                     # zero real: município sem estabelecimento com leitos
    for c in ["leitos","leitos_sus","leitos_uti","leitos_uti_sus"]:
        df[c + "_10k"] = (df[c] / df.populacao * 1e4).astype("float32")
    assert len(df) == n0, "a junção alterou o número de casos"
    relatorio = {"casos": n0, "sem_correspondencia_ibge": int(sem_ibge.sum()),
                 "municipios_sem_ibge": sorted(df.loc[sem_ibge, "cod_municipio"].dropna().astype(str).unique().tolist()),
                 "casos_em_municipio_sem_leitos_(zero_real)": int(sem_cnes.sum()), "pct_casos_em_municipio_sem_leitos": round(sem_cnes.mean()*100, 2),
                 "indicadores_saneamento": IND_SANEAMENTO}
    return df, relatorio

dev, rel_int = integrar(dev)
print(json.dumps(rel_int, indent=2, ensure_ascii=False))
NUM_CTX = ["leitos_10k","leitos_sus_10k","leitos_uti_10k","leitos_uti_sus_10k","n_estabelecimentos","densidade_hab_km2","media_moradores","populacao"] + IND_SANEAMENTO
print(dev[["populacao","densidade_hab_km2","media_moradores","n_estabelecimentos","leitos_10k","leitos_sus_10k","leitos_uti_10k","leitos_uti_sus_10k"]].describe().round(2).T.to_string())
dev["faixa_leitos"] = pd.cut(dev.leitos_10k, [-0.1, 0, 10, 20, 30, 1e9], labels=["0","0-10","10-20","20-30","30+"])
t = dev.groupby("faixa_leitos", observed=True).y.agg(casos="size", pct_positivos=lambda s: s.mean()*100).round(2)
print("\n% positivos por faixa de leitos existentes por 10 mil habitantes:"); print(t.to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(dev.leitos_10k.clip(upper=100), bins=50, color="seagreen"); ax[0].set_title("10. Leitos existentes por 10 mil hab. no município de residência"); ax[0].set_xlabel("leitos / 10 mil hab. (truncado em 100)"); ax[0].set_ylabel("Casos")
barras(ax[1], t.index, t.pct_positivos, "11. % de casos com alarme/grave por faixa de leitos", "leitos / 10 mil hab.", "% positivos", cor="seagreen", linha=dev.y.mean()*100)
plt.tight_layout(); plt.savefig(FIG/"10_11_leitos.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 12: Correlação com o alvo
# ==========================================================================================
num_cols = ["idade_anos","atraso_notif"] + SINTOMAS + COMORB + NUM_CTX
corr_alvo = dev[num_cols + ["y"]].corr()["y"].drop("y").sort_values()
print(corr_alvo.round(3).to_string())
plt.figure(figsize=(8, 8))
plt.barh(corr_alvo.index, corr_alvo.values, color=np.where(corr_alvo.values > 0, "crimson", "steelblue"))
plt.axvline(0, color="black", lw=0.8); plt.title("12. Correlação de Pearson de cada atributo com o alvo"); plt.xlabel("r")
plt.tight_layout(); plt.savefig(FIG/"12_correlacao_alvo.png", dpi=120); plt.show()
sel = ["y","idade_anos","atraso_notif","VOMITO","NAUSEA","LEUCOPENIA","PETEQUIA_N","DIABETES","HIPERTENSA","leitos_10k","leitos_uti_10k","densidade_hab_km2"]
plt.figure(figsize=(10, 8))
M = dev[sel].corr(); im = plt.imshow(M, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(im, shrink=.8)
plt.xticks(range(len(sel)), sel, rotation=90); plt.yticks(range(len(sel)), sel)
for i in range(len(sel)):
    for j in range(len(sel)): plt.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("13. Matriz de correlação (atributos selecionados)"); plt.tight_layout(); plt.savefig(FIG/"13_heatmap_corr.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 13: Preparação para o treinamento e ajuste inicial (P05 / P07)
# ==========================================================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)

BIN_SINAN = SINTOMAS + COMORB
CONT_SINAN = ["idade_anos","atraso_notif"]
CAT_SINAN = ["sexo","gestante","raca","escolaridade","uf_res","mes_sintomas"]
CONT_ENRIQ = CONT_SINAN + NUM_CTX
CAT_ENRIQ = CAT_SINAN + ["regiao"]
NUM_SINAN, NUM_ENRIQ = CONT_SINAN + BIN_SINAN, CONT_ENRIQ + BIN_SINAN
CONJUNTOS = {"SINAN isolado": (BIN_SINAN, CONT_SINAN, CAT_SINAN), "SINAN + CNES + IBGE": (BIN_SINAN, CONT_ENRIQ, CAT_ENRIQ)}

def construir_pipeline(binarias, continuas, categoricas):
    pre = ColumnTransformer([
        ("bin",  SimpleImputer(strategy="most_frequent", add_indicator=True), binarias),                               # 0/1: moda + indicador de imputação, sem padronizar
        ("cont", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)), ("escalar", StandardScaler())]), continuas),
        ("cat",  OneHotEncoder(handle_unknown="ignore", min_frequency=50), categoricas)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))])

colunas_base = ["id_registro","y","CLASSI_FIN","DT_SIN_PRI","cod_municipio","competencia","nome_municipio","regiao"] + NUM_ENRIQ + CAT_SINAN
dev = dev[colunas_base].copy()
dev.to_parquet(PROC/"base_analitica_dev.parquet", index=False)

treino, valid = train_test_split(dev, test_size=0.20, stratify=dev.y, random_state=SEED)
print(f"treino: {len(treino):,} ({treino.y.mean()*100:.2f}% pos.) | validação: {len(valid):,} ({valid.y.mean()*100:.2f}% pos.)")
def avaliar(modelo, X, y, limiar=0.5):
    p = modelo.predict_proba(X)[:, 1]
    yh = (p >= limiar).astype(int)
    return {"acuracia": accuracy_score(y, yh), "acuracia_balanceada": balanced_accuracy_score(y, yh),
            "precisao": precision_score(y, yh), "recall": recall_score(y, yh), "f1": f1_score(y, yh),
            "auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p)}, p

modelos, resultados, probs = {}, {}, {}
for nome, (b, ct, cat) in CONJUNTOS.items():
    t = time.time(); cols = b + ct + cat
    m = construir_pipeline(b, ct, cat).fit(treino[cols], treino.y)
    r, p = avaliar(m, valid[cols], valid.y)
    modelos[nome], resultados[nome], probs[nome] = m, r, p
    print(f"{nome:22s} ajustado em {time.time()-t:.0f} s | atributos após pré-processamento: {m[-1].n_features_in_}")

tabela_res = pd.DataFrame(resultados).T.round(4)
tabela_res.to_csv(PROC/"metricas_validacao.csv")
print(tabela_res)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, nome in enumerate(CONJUNTOS):
    ConfusionMatrixDisplay.from_predictions(valid.y, (probs[nome] >= 0.5).astype(int), display_labels=["dengue","alarme/grave"], colorbar=False, ax=ax[i], values_format=",d")
    ax[i].set_title(f"{14+i}. Matriz de confusão — {nome}")
for nome in CONJUNTOS:
    RocCurveDisplay.from_predictions(valid.y, probs[nome], name=nome, ax=ax[2])
ax[2].plot([0, 1], [0, 1], "k--", lw=0.8); ax[2].set_title("16. Curvas ROC na validação")
plt.tight_layout(); plt.savefig(FIG/"14_16_confusao_roc.png", dpi=120); plt.show()
def coeficientes(modelo, k=12):
    nomes = modelo["pre"].get_feature_names_out()
    co = pd.Series(modelo["clf"].coef_[0], index=[n.split("__", 1)[1] for n in nomes]).sort_values()
    return pd.concat([co.head(k), co.tail(k)])

co = coeficientes(modelos["SINAN + CNES + IBGE"])
plt.figure(figsize=(9, 7))
plt.barh(co.index, co.values, color=np.where(co.values > 0, "crimson", "steelblue"))
plt.axvline(0, color="black", lw=0.8); plt.title("17. Coeficientes da regressão logística (enriquecido): 12 menores e 12 maiores"); plt.xlabel("coeficiente (contínuas padronizadas; 0/1 e one-hot na escala original)")
plt.tight_layout(); plt.savefig(FIG/"17_coeficientes.png", dpi=120); plt.show()

# ==========================================================================================
# ETAPA 14: Como a acurácia será calculada — método de avaliação
# ==========================================================================================
for nome in CONJUNTOS:
    vn, fp, fn, vp = confusion_matrix(valid.y, (probs[nome] >= 0.5).astype(int)).ravel()
    print(f"{nome:22s} VP={vp:,} FP={fp:,} FN={fn:,} VN={vn:,} -> acurácia={(vp+vn)/(vp+vn+fp+fn):.4f} | sensibilidade={vp/(vp+fn):.4f} | especificidade={vn/(vn+fp):.4f}")

# efeito do limiar (só validação): precisão e recall para alguns cortes
p = probs["SINAN + CNES + IBGE"]
lim = pd.DataFrame([{"limiar": l, "acuracia": accuracy_score(valid.y, p >= l), "precisao": precision_score(valid.y, p >= l), "recall": recall_score(valid.y, p >= l),
                     "f1": f1_score(valid.y, p >= l), "alertas_%": (p >= l).mean()*100} for l in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]).round(3)
print(lim)
resumo_exec = {"casos_elegiveis_total": int(len(ids_dev) + len(ids_test)), "desenvolvimento": int(len(ids_dev)), "teste_reservado": int(len(ids_test)),
               "treino": int(len(treino)), "validacao": int(len(valid)), "pct_positivos": round(float(dev.y.mean()*100), 2),
               "seed": SEED, "integracao": rel_int, "metricas_validacao": {k: {m: round(v, 4) for m, v in r.items()} for k, r in resultados.items()},
               "versoes": {m.__name__: m.__version__ for m in (np, pd, sklearn, matplotlib)}}
(PROC/"resumo_integracao_modelo.json").write_text(json.dumps(resumo_exec, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(resumo_exec["metricas_validacao"], indent=2))

# ==========================================================================================
# CONCLUSÃO
# ==========================================================================================
