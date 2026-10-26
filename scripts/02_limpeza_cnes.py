#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/02_limpeza_cnes.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/02_limpeza_cnes.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# 1. Objetivo
# ==========================================================================================
# ============================================================
# 1. OBJETIVO
# ============================================================
ENTRADA_PREVISTA = "db/bases_brutas/cnes/Leitos_2025.csv"
SAIDA_PREVISTA   = "db/bases_tratadas/cnes/cnes_municipio_2025_tratado.csv"

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
warnings.filterwarnings("ignore"); plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})
pd.set_option("display.width", 160, "display.max_columns", 50)
print("pandas", pd.__version__, "| numpy", np.__version__)

# ==========================================================================================
# 3. Caminhos
# ==========================================================================================
# ============================================================
# 3. CAMINHOS
# ============================================================
RAIZ = Path(__file__).resolve().parents[1]
BRUTA, TRATADA = RAIZ/"db/bases_brutas/cnes", RAIZ/"db/bases_tratadas/cnes"
TRATADA.mkdir(parents=True, exist_ok=True)
ARQ_BRUTO   = BRUTA/"Leitos_2025.csv"
ARQ_TRATADO = TRATADA/"cnes_municipio_2025_tratado.csv"
ARQ_ESTAB   = TRATADA/"cnes_estabelecimentos_2025_validado.csv"
ARQ_LOG     = TRATADA/"cnes_limpeza_log.json"
LOG = {"entrada": str(ARQ_BRUTO.relative_to(RAIZ)), "saida": str(ARQ_TRATADO.relative_to(RAIZ)), "etapas": {}}
print("entrada existe:", ARQ_BRUTO.exists())

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
cab = open(ARQ_BRUTO, "rb").read(2048)
try: cab.decode("utf-8"); encoding = "utf-8"
except UnicodeDecodeError: encoding = "latin-1"
separador = max([",", ";", "\t"], key=lambda s: cab.split(b"\n")[0].count(s.encode()))
LOG["etapas"]["identidade"] = {"tamanho_MB": round(ARQ_BRUTO.stat().st_size/1e6, 1), "sha256": sha256(ARQ_BRUTO), "encoding": encoding, "separador": separador}
leitos = pd.read_csv(ARQ_BRUTO, sep=separador, encoding=encoding, dtype=str)
print(json.dumps(LOG["etapas"]["identidade"], indent=2)); print("shape:", leitos.shape)
leitos.head(3).T

# ==========================================================================================
# 5. Diagnóstico inicial
# ==========================================================================================
# ============================================================
# 5. DIAGNÓSTICO INICIAL
# ============================================================
COLS_LEITO = [c for c in leitos.columns if c.startswith(("LEITOS_", "UTI_"))]
diag = pd.DataFrame({"nulos_%": (leitos.isna().mean()*100).round(2), "n_distintos": leitos.nunique(), "exemplo": leitos.iloc[0].astype(str).str[:25]})
print(diag.to_string())
print("competências (COMP):", leitos.COMP.value_counts().sort_index().to_dict())
print("CO_IBGE — n dígitos:", leitos.CO_IBGE.str.len().value_counts().to_dict(), "| só dígitos:", leitos.CO_IBGE.str.fullmatch(r"\d+").all())
print("CNES — n dígitos:", leitos.CNES.str.len().value_counts().to_dict())
print("estabelecimentos distintos:", leitos.CNES.nunique(), "| municípios distintos:", leitos.CO_IBGE.nunique())
print("linhas duplicadas por (COMP, CNES):", int(leitos.duplicated(["COMP","CNES"]).sum()), "| linhas totalmente duplicadas:", int(leitos.duplicated().sum()))
print("MOTIVO_DESABILITACAO preenchido:", int(leitos.MOTIVO_DESABILITACAO.notna().sum()))
print("\nDS_TIPO_UNIDADE:"); print(leitos.DS_TIPO_UNIDADE.value_counts().to_string())
print("\nDESC_NATUREZA_JURIDICA:"); print(leitos.DESC_NATUREZA_JURIDICA.value_counts().to_string())
print("\nTP_GESTAO:", leitos.TP_GESTAO.value_counts(dropna=False).to_dict())
num = leitos[COLS_LEITO].apply(pd.to_numeric, errors="coerce")
print("valores não numéricos por coluna de leito:", (num.isna() & leitos[COLS_LEITO].notna()).sum().to_dict())
print("nulos:", num.isna().sum().to_dict())
print(num.describe().round(1).T.to_string())
print("\nConsistência entre categorias:")
print("  LEITOS_SUS > LEITOS_EXISTENTES:", int((num.LEITOS_SUS > num.LEITOS_EXISTENTES).sum()))
print("  UTI_TOTAL_EXIST > LEITOS_EXISTENTES:", int((num.UTI_TOTAL_EXIST > num.LEITOS_EXISTENTES).sum()))
print("  UTI_TOTAL_SUS > UTI_TOTAL_EXIST:", int((num.UTI_TOTAL_SUS > num.UTI_TOTAL_EXIST).sum()))
print("  UTI_TOTAL_EXIST != soma das UTIs por tipo:", int((num.UTI_TOTAL_EXIST != num[["UTI_ADULTO_EXIST","UTI_PEDIATRICO_EXIST","UTI_NEONATAL_EXIST","UTI_QUEIMADO_EXIST","UTI_CORONARIANA_EXIST"]].sum(axis=1)).sum()))
print("  valores negativos:", int((num < 0).sum().sum()), "| LEITOS_EXISTENTES = 0:", int((num.LEITOS_EXISTENTES == 0).sum()))

# ==========================================================================================
# 6. Regras de limpeza
# ==========================================================================================
# ============================================================
# 6. REGRAS DE LIMPEZA
# ============================================================
COLS_MANTER  = ["LEITOS_EXISTENTES","LEITOS_SUS","UTI_TOTAL_EXIST","UTI_TOTAL_SUS"]
COLS_UTI_TIPO = ["UTI_ADULTO_EXIST","UTI_ADULTO_SUS","UTI_PEDIATRICO_EXIST","UTI_PEDIATRICO_SUS","UTI_NEONATAL_EXIST","UTI_NEONATAL_SUS","UTI_QUEIMADO_EXIST","UTI_QUEIMADO_SUS","UTI_CORONARIANA_EXIST","UTI_CORONARIANA_SUS"]
COLS_DESCARTAR = ["NO_LOGRADOURO","NU_ENDERECO","NO_COMPLEMENTO","NO_BAIRRO","CO_CEP","NU_TELEFONE","NO_EMAIL","RAZAO_SOCIAL","NOME_ESTABELECIMENTO","REGIAO","UF","MUNICIPIO","MOTIVO_DESABILITACAO"]
RENOMEAR = {"LEITOS_EXISTENTES": "leitos", "LEITOS_SUS": "leitos_sus", "UTI_TOTAL_EXIST": "leitos_uti", "UTI_TOTAL_SUS": "leitos_uti_sus"}
print("mantidas:", COLS_MANTER, "\nno arquivo de estabelecimentos também:", COLS_UTI_TIPO, "\ndescartadas:", COLS_DESCARTAR)

# ==========================================================================================
# 7. Aplicação
# ==========================================================================================
# ============================================================
# 7. APLICAÇÃO
# ============================================================
est = leitos.drop(columns=COLS_DESCARTAR).copy()
est["competencia"] = est.COMP.astype(int)                                              # R1
est["cod_municipio"] = est.CO_IBGE.str.strip().str.zfill(6)                            # R2
for c in COLS_MANTER + COLS_UTI_TIPO: est[c] = pd.to_numeric(est[c], errors="coerce").astype("Int64")   # R1
est["flag_inconsistencia"] = ((est.LEITOS_SUS > est.LEITOS_EXISTENTES) | (est.UTI_TOTAL_EXIST > est.LEITOS_EXISTENTES) | (est.UTI_TOTAL_SUS > est.UTI_TOTAL_EXIST)).astype(int)  # R3
n_incons = int(est.flag_inconsistencia.sum()); n_nulos = int(est[COLS_MANTER].isna().sum().sum())
print("linhas com inconsistência entre categorias (marcadas):", n_incons, "| nulos nas colunas de leito:", n_nulos)

cnes_mun = (est.groupby(["cod_municipio","competencia"])                               # R5
            .agg(n_estabelecimentos=("CNES","nunique"), leitos=("LEITOS_EXISTENTES","sum"), leitos_sus=("LEITOS_SUS","sum"),
                 leitos_uti=("UTI_TOTAL_EXIST","sum"), leitos_uti_sus=("UTI_TOTAL_SUS","sum"))
            .reset_index())
LOG["etapas"]["aplicacao"] = {"linhas_estabelecimento": int(len(est)), "inconsistencias_marcadas": n_incons, "nulos_leito": n_nulos,
                              "linhas_municipio_competencia": int(len(cnes_mun)), "municipios": int(cnes_mun.cod_municipio.nunique())}
print(json.dumps(LOG["etapas"]["aplicacao"], indent=2))
print(cnes_mun.head())

# ==========================================================================================
# 8. Validação
# ==========================================================================================
# ============================================================
# 8. VALIDAÇÃO
# ============================================================
checks = {
 "chave (cod_municipio, competencia) unica": bool(not cnes_mun.duplicated(["cod_municipio","competencia"]).any()),
 "soma de leitos preservada": bool(cnes_mun.leitos.sum() == est.LEITOS_EXISTENTES.sum()),
 "soma de UTI preservada": bool(cnes_mun.leitos_uti.sum() == est.UTI_TOTAL_EXIST.sum()),
 "12 competencias": bool(cnes_mun.competencia.nunique() == 12),
 "cod_municipio 6 digitos": bool(cnes_mun.cod_municipio.str.len().eq(6).all()),
 "leitos_sus <= leitos (municipio)": bool((cnes_mun.leitos_sus <= cnes_mun.leitos).all()),
 "sem negativos": bool((cnes_mun[["leitos","leitos_sus","leitos_uti","leitos_uti_sus"]] >= 0).all().all()),
}
print(pd.Series(checks).to_string()); assert all(checks.values()); LOG["etapas"]["validacao"] = checks
print("\nmunicípios por competência:", cnes_mun.groupby("competencia").cod_municipio.nunique().to_dict())
print("total de leitos por competência:", cnes_mun.groupby("competencia").leitos.sum().to_dict())
dez = cnes_mun[cnes_mun.competencia == 202512]
print("Distribuição por município (dez/2025):"); print(dez[["n_estabelecimentos","leitos","leitos_sus","leitos_uti"]].describe().round(1).T.to_string())
print("\nmunicípios com leitos cadastrados em dez/2025:", len(dez), "de 5.570 no Censo 2022 ->", f"{5570-len(dez):,} municípios sem estabelecimento com leitos (zero real na integração)")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(dez.leitos.clip(upper=500), bins=50, color="seagreen"); ax[0].set_title("Leitos por município, dez/2025 (truncado em 500)"); ax[0].set_xlabel("leitos")
serie = cnes_mun.groupby("competencia")[["leitos","leitos_uti"]].sum()
ax[1].plot(range(12), serie.leitos/1e3, "o-", label="leitos (mil)"); ax[1].plot(range(12), serie.leitos_uti/1e3, "s-", label="UTI (mil)"); ax[1].set_xticks(range(12)); ax[1].set_xticklabels([str(c)[-2:] for c in serie.index]); ax[1].set_title("Total nacional por competência de 2025"); ax[1].legend()
plt.tight_layout(); plt.show()

# ==========================================================================================
# 9. Exportação
# ==========================================================================================
# ============================================================
# 9. EXPORTAÇÃO
# ============================================================
cnes_mun.to_csv(ARQ_TRATADO, index=False)
est.rename(columns=RENOMEAR)[["competencia","cod_municipio","CNES","TP_GESTAO","CO_TIPO_UNIDADE","DS_TIPO_UNIDADE","NATUREZA_JURIDICA","DESC_NATUREZA_JURIDICA"] + list(RENOMEAR.values()) + COLS_UTI_TIPO + ["flag_inconsistencia"]].to_csv(ARQ_ESTAB, index=False)
LOG["etapas"]["exportacao"] = {"linhas": int(len(cnes_mun)), "colunas": list(cnes_mun.columns), "tamanho_KB": round(ARQ_TRATADO.stat().st_size/1e3)}
LOG["executado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
ARQ_LOG.write_text(json.dumps(LOG, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(LOG["etapas"]["exportacao"], indent=2))

# ==========================================================================================
# 10. Resumo
# ==========================================================================================
# ============================================================
# 10. RESUMO
# ============================================================
pd.DataFrame([
    ["linhas no bruto (estabelecimento x competência)", len(leitos), ""],
    ["linhas totalmente duplicadas", int(leitos.duplicated().sum()), "nenhuma removida"],
    ["colunas no bruto / descartadas / mantidas", f"{leitos.shape[1]} / {len(COLS_DESCARTAR)} / {leitos.shape[1]-len(COLS_DESCARTAR)}", "R6"],
    ["inconsistências entre categorias (marcadas)", n_incons, "R3"],
    ["nulos nas colunas de leito", n_nulos, "R4"],
    ["linhas na saída (município x competência)", len(cnes_mun), "R5"],
    ["municípios com leitos", cnes_mun.cod_municipio.nunique(), "os demais = zero real"],
    ["competências", cnes_mun.competencia.nunique(), "202501-202512"],
], columns=["item","valor","observação"])
