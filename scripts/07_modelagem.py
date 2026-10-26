#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/07_modelagem_avaliacao.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/07_modelagem_avaliacao.py
Pré-requisito: scripts 01 a 05 já executados. A avaliação no teste (ETAPA 5) deve rodar uma única vez."""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# ETAPA 1: Bibliotecas, caminhos e variáveis
# ==========================================================================================

import json, gc, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay)

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})
pd.set_option("display.width", 160, "display.max_columns", 40)
SEED = 42

RAIZ = Path(__file__).resolve().parent.parent
TRATADA = RAIZ/"db/bases_tratadas"
PROC, FIG = TRATADA/"analitica", RAIZ/"reports/figuras"
FIG.mkdir(parents=True, exist_ok=True)

# bases municipais (usadas para montar o conjunto de teste, como no notebook 05)
ibge = pd.read_csv(TRATADA/"ibge/ibge_municipio_2022_tratado.csv", dtype={"cod_municipio": str, "cd_mun_ibge7": str})
cnes = pd.read_csv(TRATADA/"cnes/cnes_municipio_2025_tratado.csv", dtype={"cod_municipio": str})

SINTOMAS = ["FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS","CONJUNTVIT","ARTRITE","ARTRALGIA","PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO"]
COMORB   = ["DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
NUM_CTX  = ["leitos_10k","leitos_sus_10k","leitos_uti_10k","leitos_uti_sus_10k","n_estabelecimentos","densidade_hab_km2","media_moradores","populacao"]

BINARIAS    = SINTOMAS + COMORB                                    # 0/1
CONTINUAS   = ["idade_anos", "atraso_notif"] + NUM_CTX             # números
CATEGORICAS = ["sexo","gestante","raca","escolaridade","uf_res","mes_sintomas","regiao"]
COLUNAS     = BINARIAS + CONTINUAS + CATEGORICAS
print(len(COLUNAS), "atributos de entrada")


# ==========================================================================================
# ETAPA 2: Dados de desenvolvimento e a mesma divisão da Etapa 2
# ==========================================================================================

dev = pd.read_parquet(PROC/"base_analitica_dev.parquet")
treino, valid = train_test_split(dev, test_size=0.20, stratify=dev.y, random_state=SEED)
print(f"treino: {len(treino):,} casos | validação: {len(valid):,} casos | % de graves no treino: {treino.y.mean()*100:.2f}%")
print("no relatório da Etapa 2: treino 948.352 e validação 237.089")


# ==========================================================================================
# ETAPA 3: Os dois métodos
# ==========================================================================================

def construir_pipeline(classificador):
    pre = ColumnTransformer([
        ("bin",  SimpleImputer(strategy="most_frequent", add_indicator=True), BINARIAS),
        ("cont", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)), ("escalar", StandardScaler())]), CONTINUAS),
        ("cat",  OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAS)])
    return Pipeline([("pre", pre), ("clf", classificador)])

modelos = {
    "Regressão Logística": construir_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": construir_pipeline(RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50, max_samples=0.3,
                                                              class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
}
probs_valid = {}
for nome, m in modelos.items():
    t0 = time.time()
    m.fit(treino[COLUNAS], treino.y)                                   # o Random Forest pode levar alguns minutos
    probs_valid[nome] = m.predict_proba(valid[COLUNAS])[:, 1]
    print(f"{nome}: treinado em {time.time()-t0:.0f} s")


# ==========================================================================================
# ETAPA 4: Escolha do modelo e do limiar (só na validação)
# ==========================================================================================

RECALL_ALVO = 0.70      # decisão do grupo: "queremos encontrar pelo menos 70% dos casos graves"

def escolher_limiar(y, p, recall_alvo):
    prec, rec, lim = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall_alvo)[0]          # limiares que cumprem a meta
    return float(lim[ok[-1]])                           # o maior deles

def metricas(y, p, limiar):
    y = np.asarray(y); yh = (np.asarray(p) >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"recall": recall_score(y, yh), "precisao": precision_score(y, yh, zero_division=0), "f1": f1_score(y, yh),
            "auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p), "acuracia": accuracy_score(y, yh),
            "alertas_pct": yh.mean(), "vp": int(vp), "fp": int(fp), "fn": int(fn), "vn": int(vn)}

linhas = {}
for nome, p in probs_valid.items():
    lim = escolher_limiar(valid.y, p, RECALL_ALVO)
    linhas[nome] = {"limiar": lim, **metricas(valid.y, p, lim)}
tab_valid = pd.DataFrame(linhas).T[["limiar","recall","precisao","f1","auc_roc","auc_pr","alertas_pct"]].round(4)
print(f"VALIDAÇÃO — limiar escolhido para recall ≥ {RECALL_ALVO:.0%}:"); print(tab_valid.to_string())

escolhido = tab_valid.auc_pr.idxmax()
LIMIAR = float(tab_valid.loc[escolhido, "limiar"])
modelo_final, p_valid = modelos[escolhido], probs_valid[escolhido]
print(f"\nModelo escolhido (maior AUC-PR): {escolhido} | limiar: {LIMIAR:.4f}")

# O que cada meta de recall custa? (modelo escolhido, validação)
prev = valid.y.mean(); comp = []
for alvo in [0.5, 0.6, 0.7, 0.8, 0.9]:
    lim = escolher_limiar(valid.y, p_valid, alvo); m = metricas(valid.y, p_valid, lim)
    comp.append({"recall_alvo": alvo, "limiar": lim, "precisao": m["precisao"], "casos_alertados_%": m["alertas_pct"]*100})
comp = pd.DataFrame(comp).round(3); print(comp.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for nome, p in probs_valid.items(): RocCurveDisplay.from_predictions(valid.y, p, name=nome, ax=ax[0])
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_title("Curva ROC na validação")
ax[1].plot(comp.recall_alvo*100, comp["casos_alertados_%"], "o-", color="crimson")
ax[1].set_xlabel("meta de recall (%)"); ax[1].set_ylabel("% de casos alertados"); ax[1].set_title("Quanto mais graves capturados, mais alertas")
plt.tight_layout(); plt.savefig(FIG/"20_validacao_modelos_e_limiar.png", dpi=120); plt.show()


# ==========================================================================================
# ETAPA 5: Avaliação única no conjunto de teste
# ==========================================================================================

def integrar(df):
    """Mesma função do notebook 05: junta IBGE e CNES ao caso pelo código do município."""
    n0 = len(df)
    df = df.merge(ibge.drop(columns=["cd_mun_ibge7","uf","regiao"]), on="cod_municipio", how="left", validate="m:1")
    df["competencia_cnes"] = df.competencia.clip(lower=202501, upper=202512)
    df = df.merge(cnes.rename(columns={"competencia": "competencia_cnes"}), on=["cod_municipio","competencia_cnes"], how="left", validate="m:1")
    for c in ["n_estabelecimentos","leitos","leitos_sus","leitos_uti","leitos_uti_sus"]: df[c] = df[c].fillna(0).astype("float32")
    for c in ["leitos","leitos_sus","leitos_uti","leitos_uti_sus"]: df[c + "_10k"] = (df[c] / df.populacao * 1e4).astype("float32")
    assert len(df) == n0
    return df

ids_test = pd.read_csv(PROC/"split_ids.csv").query("conjunto == 'teste'").id_registro.values
sinan = pd.read_parquet(TRATADA/"sinan/sinan_dengue_2025_tratado.parquet")
for c in ["sexo","gestante","raca","escolaridade","uf_res","cod_municipio"]: sinan[c] = sinan[c].astype("category")
teste = sinan[sinan.id_registro.isin(ids_test)].copy(); del sinan; gc.collect()

teste = teste.rename(columns={"dias_ate_notificacao": "atraso_notif"}); teste["y"] = teste.alvo
for c in SINTOMAS + COMORB: teste[c] = teste[c + "_bin"]
teste = teste.merge(ibge[["cod_municipio","regiao"]], on="cod_municipio", how="left")
teste["regiao"] = teste.regiao.fillna("ignorado").astype("category")
teste = integrar(teste)

assert len(teste) == len(ids_test) and not (set(ids_test) & set(dev.id_registro)), "o teste tem de ser separado do desenvolvimento"
print(f"teste: {len(teste):,} casos | graves: {int(teste.y.sum()):,} ({teste.y.mean()*100:.2f}%)")

p_teste = modelo_final.predict_proba(teste[COLUNAS])[:, 1]        # única passagem do teste pelo modelo
y_teste = teste.y.values.astype(int)
r = metricas(y_teste, p_teste, LIMIAR)

comparacao = pd.DataFrame({"validação": metricas(valid.y, p_valid, LIMIAR), "teste": r}).loc[["recall","precisao","f1","auc_pr","auc_roc","acuracia","alertas_pct"]].round(4)
print(f"Modelo: {escolhido} | limiar: {LIMIAR:.4f}\n"); print(comparacao.to_string())
print(f"\nMatriz de confusão no teste: VP={r['vp']:,}  FP={r['fp']:,}  FN={r['fn']:,}  VN={r['vn']:,}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ConfusionMatrixDisplay.from_predictions(y_teste, (p_teste >= LIMIAR).astype(int), display_labels=["dengue","alarme/grave"], colorbar=False, ax=ax[0], values_format=",d")
ax[0].set_title("Matriz de confusão no teste")
RocCurveDisplay.from_predictions(y_teste, p_teste, name=f"{escolhido} (AUC={r['auc_roc']:.3f})", ax=ax[1]); ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_title("Curva ROC no teste")
plt.tight_layout(); plt.savefig(FIG/"21_teste_confusao_roc.png", dpi=120); plt.show()


# ==========================================================================================
# ETAPA 6: Resultado preliminar e produto
# ==========================================================================================

teste["escore"] = p_teste
teste["faixa"] = np.where(p_teste >= LIMIAR, "alto risco", "baixo risco")
graves_total = int(teste.y.sum())
faixas = teste.groupby("faixa").agg(casos=("y", "size"), graves=("y", "sum"))
faixas["% dos casos"] = faixas.casos / len(teste) * 100
faixas["% de graves na faixa"] = faixas.graves / faixas.casos * 100
faixas["% dos graves capturados"] = faixas.graves / graves_total * 100
print("Faixas de risco no TESTE:"); print(faixas.round(2).to_string())

por_mil = {"graves_esperados": 1000*y_teste.mean(), "alertas": 1000*r["alertas_pct"], "graves_capturados": 1000*r["vp"]/len(y_teste)}
print("\nPor 1.000 notificações:", {k: round(float(v), 1) for k, v in por_mil.items()})

# exemplo da saída do produto: 4 casos de cada faixa
exemplo = pd.concat([d.sample(4, random_state=SEED) for _, d in teste.groupby("faixa")])
exemplo = exemplo[["id_registro","idade_anos","VOMITO","LEUCOPENIA","HIPERTENSA","DIABETES","escore","faixa","y"]].copy()
exemplo["escore"] = exemplo.escore.round(2); exemplo["idade_anos"] = exemplo.idade_anos.round(0)
exemplo = exemplo.rename(columns={"y": "evolucao_real (1=alarme/grave)"})
print("\nExemplo da saída do produto (casos do teste):"); print(exemplo.to_string(index=False))
exemplo.to_csv(PROC/"etapa3_exemplo_predicoes.csv", index=False)


# ==========================================================================================
# ETAPA 7: Números para o storytelling e resumo
# ==========================================================================================

arq = TRATADA/"informes/informes_indicadores_citados.csv"
if arq.exists():
    ult = pd.read_csv(arq).sort_values("informe_num").iloc[-1]
    print(f"Informe nº {int(ult.informe_num)} (SE 1 a {int(ult.se_fim)}): {ult.casos_provaveis:,.0f} casos prováveis | incidência {ult.incidencia_100mil} por 100 mil | "
          f"variação vs 2024: {ult.variacao_vs_2024_pct}% | óbitos confirmados: {ult.obitos_confirmados:,.0f} | em investigação: {ult.obitos_investigacao:,.0f} | sorotipo predominante: {ult.sorotipo_predominante}")
else:
    print("Rode o notebook 06 para preencher o contexto dos informes.")

def pct(x, c=1): return f"{x*100:.{c}f}%".replace(".", ",")
def num(x, c=3): return f"{x:.{c}f}".replace(".", ",")
def mil(x): return f"{int(round(x)):,}".replace(",", ".")

print("=== CAMPOS PARA O RELATÓRIO (copie o valor ao lado do nome) ===")
campos = {
  "modelo_escolhido": escolhido,
  "recall_alvo": pct(RECALL_ALVO, 0),
  "limiar": num(LIMIAR, 3),
  "n_treino / n_validacao": f"{mil(len(treino))} / {mil(len(valid))}",
  "n_teste / graves_teste": f"{mil(len(y_teste))} / {mil(y_teste.sum())} ({pct(y_teste.mean(), 2)})",
  "recall_teste": pct(r["recall"]), "precisao_teste": pct(r["precisao"]), "f1_teste": num(r["f1"]),
  "auc_pr_teste": num(r["auc_pr"]), "auc_roc_teste": num(r["auc_roc"]), "acuracia_teste": pct(r["acuracia"]),
  "alertas_teste": pct(r["alertas_pct"]),
  "VP / FP / FN / VN": " / ".join(mil(r[k]) for k in ["vp","fp","fn","vn"]),
  "por_mil (graves esperados / alertas / graves capturados)": f"{num(por_mil['graves_esperados'],1)} / {num(por_mil['alertas'],0)} / {num(por_mil['graves_capturados'],1)}",
}
for k, v in campos.items(): print(f"{k:58s} {v}")

print("\n=== TABELA DE VALIDAÇÃO (modelos comparados) ==="); print(tab_valid.to_string())
print("\n=== FAIXAS DE RISCO (teste) ==="); print(faixas.round(1).to_string())
(PROC/"resumo_etapa3.json").write_text(json.dumps({"campos": campos, "validacao": tab_valid.reset_index().to_dict(orient="records")}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
