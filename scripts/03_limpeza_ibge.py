#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/03_limpeza_ibge.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/03_limpeza_ibge.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# 1. Objetivo
# ==========================================================================================
# ============================================================
# 1. OBJETIVO
# ============================================================
ENTRADAS_PREVISTAS = ["Agregados_por_municipios_basico_BR.csv", "Agregados_por_municipios_caracteristicas_domicilio1_BR.csv",
                      "Agregados_por_municipios_caracteristicas_domicilio2_BR.csv", "Agregados_por_municipios_caracteristicas_domicilio3_BR.csv"]
SAIDA_PREVISTA = "db/bases_tratadas/ibge/ibge_municipio_2022_tratado.csv"

# ==========================================================================================
# 2. Bibliotecas
# ==========================================================================================
# ============================================================
# 2. BIBLIOTECAS
# ============================================================
import json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import openpyxl; OPENPYXL = openpyxl.__version__
except ImportError:
    OPENPYXL = None
warnings.filterwarnings("ignore"); plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})
pd.set_option("display.width", 160, "display.max_columns", 40)
print("pandas", pd.__version__, "| numpy", np.__version__, "| openpyxl", OPENPYXL)

# ==========================================================================================
# 3. Caminhos
# ==========================================================================================
# ============================================================
# 3. CAMINHOS
# ============================================================
RAIZ = Path(__file__).resolve().parents[1]
BRUTA, TRATADA = RAIZ/"db/bases_brutas/ibge", RAIZ/"db/bases_tratadas/ibge"
TRATADA.mkdir(parents=True, exist_ok=True)
ARQ = {"basico": BRUTA/ENTRADAS_PREVISTAS[0], "domicilio1": BRUTA/ENTRADAS_PREVISTAS[1], "domicilio2": BRUTA/ENTRADAS_PREVISTAS[2], "domicilio3": BRUTA/ENTRADAS_PREVISTAS[3]}
ARQ_MAPEAMENTO = BRUTA/"dicionario/mapeamento_indicadores.csv"
ARQ_TRATADO    = TRATADA/"ibge_municipio_2022_tratado.csv"
ARQ_LOG        = TRATADA/"ibge_limpeza_log.json"
SINAN_TRATADO  = RAIZ/"db/bases_tratadas/sinan/sinan_dengue_2025_tratado.parquet"   # só para validar a chave
LOG = {"entradas": [str(p.relative_to(RAIZ)) for p in ARQ.values()], "saida": str(ARQ_TRATADO.relative_to(RAIZ)), "etapas": {}}
for k, p in ARQ.items(): print(f"{k:11s} existe: {p.exists()}")
print("mapeamento do dicionário existe:", ARQ_MAPEAMENTO.exists())

# ==========================================================================================
# 4. Carregamento
# ==========================================================================================
# ============================================================
# 4. CARREGAMENTO
# ============================================================
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parte in iter(lambda: f.read(1 << 24), b""): h.update(parte)
    return h.hexdigest()
ibge = {k: pd.read_csv(p, sep=";", encoding="latin1", dtype=str) for k, p in ARQ.items()}
LOG["etapas"]["identidade"] = {k: {"tamanho_MB": round(p.stat().st_size/1e6, 1), "sha256": sha256(p), "shape": list(ibge[k].shape)} for k, p in ARQ.items()}
for k, d in ibge.items(): print(f"{k:11s} {d.shape} | colunas {d.columns[0]} … {d.columns[-1]}")
ibge["basico"].head(3).T

# ==========================================================================================
# 5. Diagnóstico inicial
# ==========================================================================================
# ============================================================
# 5. DIAGNÓSTICO INICIAL
# ============================================================
for k, d in ibge.items():
    cols_num = [c for c in d.columns if c.lower().startswith("v") or c == "AREA_KM2"]      # variáveis numéricas; nomes e códigos de recorte ficam de fora
    vals = d[cols_num].values.ravel()
    especiais = pd.Series(vals[~pd.Series(vals).str.fullmatch(r"-?\d+([,.]\d+)?").fillna(False).values]).value_counts().head(5).to_dict()
    print(f"{k:11s} CD_MUN único: {d.CD_MUN.is_unique} | dígitos: {d.CD_MUN.str.len().unique()} | variáveis numéricas: {len(cols_num)} | nulos nelas: {int(d[cols_num].isna().sum().sum())} | marcadores não numéricos: {especiais or 'nenhum'}")
b = ibge["basico"]
recortes = ["CD_NU","NM_NU","CD_AGLOM","NM_AGLOM","CD_RGINT","NM_RGINT","CD_RGI","NM_RGI","CD_CONCURB","NM_CONCURB"]
print("\nbásico — recortes territoriais (não usados): campos com '.' ou vazio =", {c: int(((b[c] == ".") | b[c].isna()).sum()) for c in recortes})
b = ibge["basico"]
num_b = b[["AREA_KM2","v0001","v0002","v0003","v0004","v0005","v0006","v0007","v0008","v0009"]].apply(lambda s: pd.to_numeric(s.str.replace(",", "."), errors="coerce"))
print(num_b.describe().round(2).T.to_string())
sp = b[b.CD_MUN == "3550308"].iloc[0]
print("\nValidação — São Paulo:", {c: sp[c] for c in ["NM_MUN","NM_UF","NM_REGIAO","AREA_KM2","v0001","v0005","v0007"]})
print("soma de v0001 (população residente, Brasil):", f"{int(num_b.v0001.sum()):,}")
print("municípios com AREA_KM2 nula ou zero:", int((num_b.AREA_KM2.isna() | (num_b.AREA_KM2 <= 0)).sum()))
print("\nRegiões:", b.NM_REGIAO.value_counts().to_dict())

# ==========================================================================================
# 6. Regras de limpeza
# ==========================================================================================
# ============================================================
# 6. REGRAS DE LIMPEZA
# ============================================================
VARS_BASICO = {"v0001": "populacao", "AREA_KM2": "area_km2", "v0007": "domicilios_ocupados", "v0005": "media_moradores"}
mapeamento = pd.read_csv(ARQ_MAPEAMENTO, dtype=str) if ARQ_MAPEAMENTO.exists() else pd.DataFrame(columns=["indicador","arquivo","numerador","denominador","descricao","aprovado"])
aprovados = mapeamento[mapeamento.aprovado.str.lower() == "sim"].copy()
print("linhas no mapeamento:", len(mapeamento), "| aprovadas:", len(aprovados))
print(mapeamento)

# ==========================================================================================
# 7. Aplicação
# ==========================================================================================
# ============================================================
# 7. APLICAÇÃO
# ============================================================
def numero(s): return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
b = ibge["basico"]
saida = pd.DataFrame({"cod_municipio": b.CD_MUN.str[:6], "cd_mun_ibge7": b.CD_MUN, "nome_municipio": b.NM_MUN, "uf": b.NM_UF, "regiao": b.NM_REGIAO})
for v, nome in VARS_BASICO.items(): saida[nome] = numero(b[v])                          # R2
saida["densidade_hab_km2"] = (saida.populacao / saida.area_km2).round(2)

indicadores_calculados, problemas = [], []                                                  # R3
for _, r in aprovados.iterrows():
    d = ibge.get(r.arquivo)
    if d is None or r.numerador not in d.columns or r.denominador not in d.columns:
        problemas.append(f"{r.indicador}: arquivo/códigos não encontrados ({r.arquivo}, {r.numerador}, {r.denominador})"); continue
    num, den = numero(d[r.numerador]), numero(d[r.denominador])
    if (den <= 0).any() or den.isna().any(): problemas.append(f"{r.indicador}: denominador nulo ou zero em {int(((den <= 0) | den.isna()).sum())} municípios")
    s = pd.DataFrame({"cod_municipio": d.CD_MUN.str[:6], r.indicador: (num / den * 100).round(2)})
    saida = saida.merge(s, on="cod_municipio", how="left"); indicadores_calculados.append(r.indicador)
print("indicadores domiciliares calculados:", indicadores_calculados or "nenhum (mapeamento do dicionário ainda não aprovado)")
if problemas: print("problemas:", *problemas, sep="\n  ")
LOG["etapas"]["aplicacao"] = {"variaveis_basico": VARS_BASICO, "indicadores_domiciliares": indicadores_calculados, "problemas": problemas}
print(saida.head())

# ==========================================================================================
# 8. Validação
# ==========================================================================================
# ============================================================
# 8. VALIDAÇÃO
# ============================================================
checks = {
 "uma linha por cod_municipio": bool(saida.cod_municipio.is_unique),
 "5570 municipios": bool(len(saida) == 5570),
 "cod_municipio 6 digitos": bool(saida.cod_municipio.str.len().eq(6).all()),
 "sem nulos em populacao/area/domicilios/media": bool(saida[list(VARS_BASICO.values())].notna().all().all()),
 "populacao > 0": bool((saida.populacao > 0).all()),
 "densidade finita": bool(np.isfinite(saida.densidade_hab_km2).all()),
}
print(pd.Series(checks).to_string()); assert all(checks.values()); LOG["etapas"]["validacao"] = checks
print(saida[["populacao","area_km2","domicilios_ocupados","media_moradores","densidade_hab_km2"]].describe().round(1).T.to_string())
# R5 — compatibilidade com a chave do SINAN tratado
if SINAN_TRATADO.exists():
    chaves_sinan = pd.read_parquet(SINAN_TRATADO, columns=["cod_municipio"]).cod_municipio
    presentes = chaves_sinan.isin(saida.cod_municipio)
    sem = chaves_sinan[~presentes & chaves_sinan.notna()].astype(str).value_counts()
    compat = {"casos_sinan": int(len(chaves_sinan)), "casos_com_municipio_no_ibge": int(presentes.sum()), "casos_sem_correspondencia": int((~presentes).sum()),
              "casos_com_codigo_nulo": int(chaves_sinan.isna().sum()), "codigos_sem_correspondencia": sem.to_dict(),
              "municipios_do_sinan_cobertos": int(chaves_sinan.dropna().nunique() - len(sem)), "municipios_do_sinan": int(chaves_sinan.dropna().nunique())}
    print(json.dumps(compat, indent=2)); LOG["etapas"]["compatibilidade_sinan"] = compat
else:
    print("base tratada do SINAN não encontrada — execute 01_limpeza_sinan.ipynb antes para validar a chave")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(np.log10(saida.populacao), bins=40, color="steelblue"); ax[0].set_title("População municipal (log10)"); ax[0].set_xlabel("log10(habitantes)")
regs = sorted(saida.regiao.unique()); ax[1].boxplot([saida.loc[saida.regiao == r, "media_moradores"] for r in regs], tick_labels=regs, showfliers=False); ax[1].set_title("Média de moradores por domicílio, por região")
plt.tight_layout(); plt.show()

# ==========================================================================================
# 9. Exportação
# ==========================================================================================
# ============================================================
# 9. EXPORTAÇÃO
# ============================================================
saida.to_csv(ARQ_TRATADO, index=False)
LOG["etapas"]["exportacao"] = {"linhas": int(len(saida)), "colunas": list(saida.columns), "tamanho_KB": round(ARQ_TRATADO.stat().st_size/1e3)}
LOG["executado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
ARQ_LOG.write_text(json.dumps(LOG, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(LOG["etapas"]["exportacao"], indent=2, ensure_ascii=False))

# ==========================================================================================
# 10. Resumo
# ==========================================================================================
# ============================================================
# 10. RESUMO
# ============================================================
pd.DataFrame([
    ["arquivos de entrada", 4, "básico + domicílios 1, 2, 3"],
    ["municípios (linhas) em cada entrada", 5570, "CD_MUN único, 7 dígitos"],
    ["variáveis disponíveis nos arquivos de domicílio", 643, "V00001–V00643, codificadas"],
    ["variáveis do básico usadas", len(VARS_BASICO), ", ".join(VARS_BASICO)],
    ["indicadores domiciliares calculados", len(indicadores_calculados), "dependem do mapeamento aprovado"],
    ["linhas na saída", len(saida), "uma por município"],
    ["colunas na saída", saida.shape[1], ""],
    ["casos do SINAN com município no IBGE", LOG["etapas"].get("compatibilidade_sinan", {}).get("casos_com_municipio_no_ibge", "n/d"), "R5"],
], columns=["item","valor","observação"])
