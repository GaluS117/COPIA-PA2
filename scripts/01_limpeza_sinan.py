#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/01_limpeza_sinan.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/01_limpeza_sinan.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# 1. Objetivo
# ==========================================================================================
# ============================================================
# 1. OBJETIVO
# ============================================================
ENTRADA_PREVISTA = "db/bases_brutas/sinan/dados/DENGBR25.csv"
SAIDA_PREVISTA   = "db/bases_tratadas/sinan/sinan_dengue_2025_tratado.csv"

# ==========================================================================================
# 2. Bibliotecas
# ==========================================================================================
# ============================================================
# 2. BIBLIOTECAS
# ============================================================
import sys, platform, json, hashlib, gc, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})
pd.set_option("display.width", 160, "display.max_columns", 60, "display.max_rows", 150)
print("Python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)

# ==========================================================================================
# 3. Caminhos
# ==========================================================================================
# ============================================================
# 3. CAMINHOS
# ============================================================
RAIZ = Path(__file__).resolve().parents[1]
BRUTA   = RAIZ/"db/bases_brutas/sinan/dados"
TRATADA = RAIZ/"db/bases_tratadas/sinan"
TRATADA.mkdir(parents=True, exist_ok=True)

ARQ_BRUTO          = BRUTA/"DENGBR25.csv"
ARQ_TRATADO        = TRATADA/"sinan_dengue_2025_tratado.csv"
ARQ_AUDITORIA      = TRATADA/"sinan_dengue_2025_auditoria.csv"
ARQ_DECISAO_COLS   = TRATADA/"sinan_colunas_decisao.csv"
ARQ_DICIONARIO     = TRATADA/"sinan_dengue_2025_tratado_dicionario.csv"
ARQ_LOG            = TRATADA/"sinan_limpeza_log.json"
LOG = {"entrada": str(ARQ_BRUTO.relative_to(RAIZ)), "saida": str(ARQ_TRATADO.relative_to(RAIZ)), "etapas": {}}
print("entrada:", ARQ_BRUTO.relative_to(RAIZ), "| existe:", ARQ_BRUTO.exists())

# ==========================================================================================
# 4. Carregamento
# ==========================================================================================
# ============================================================
# 4. CARREGAMENTO
# ============================================================
def sha256(caminho, bloco=1 << 24):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for parte in iter(lambda: f.read(bloco), b""): h.update(parte)
    return h.hexdigest()

t0 = time.time()
cabecalho = open(ARQ_BRUTO, "rb").read(4096)
bom = cabecalho.startswith(b"\xef\xbb\xbf")
try:
    cabecalho.decode("utf-8"); encoding = "utf-8-sig" if bom else "utf-8"
except UnicodeDecodeError:
    encoding = "latin-1"
primeira = cabecalho.split(b"\n")[0]
separador = max([",", ";", "\t", "|"], key=lambda s: primeira.count(s.encode()))
LOG["etapas"]["identidade"] = {"tamanho_MB": round(ARQ_BRUTO.stat().st_size/1e6, 1), "sha256": sha256(ARQ_BRUTO),
                               "encoding": encoding, "separador": separador, "bom": bom}
print(json.dumps(LOG["etapas"]["identidade"], indent=2), f"\n({time.time()-t0:.0f} s)")
amostra = pd.read_csv(ARQ_BRUTO, sep=separador, encoding=encoding, nrows=200_000, low_memory=False)
print("amostra:", amostra.shape)
tipos_inferidos = pd.DataFrame({"dtype_inferido": amostra.dtypes.astype(str),
                                "nulos_%_amostra": (amostra.isna().mean()*100).round(1),
                                "n_distintos_amostra": amostra.nunique(),
                                "exemplo": amostra.iloc[0].astype(str).str[:12]})
print(tipos_inferidos)

# ==========================================================================================
# 5. Diagnóstico inicial
# ==========================================================================================
# ============================================================
# 5. DIAGNÓSTICO INICIAL
# ============================================================
t0 = time.time()
COLS_CODIGO = ["CLASSI_FIN","CRITERIO","EVOLUCAO","HOSPITALIZ","TPAUTOCTO","CS_SEXO","CS_GESTANT","CS_RACA","CS_ESCOL_N","TP_NOT","ID_AGRAVO","NU_ANO",
               "FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS","CONJUNTVIT","ARTRITE","ARTRALGIA","PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO",
               "DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
nulos = None; total = 0; codigos = {c: Counter() for c in COLS_CODIGO}
datas_min, datas_max = {}, {}
for bloco in pd.read_csv(ARQ_BRUTO, sep=separador, encoding=encoding, dtype=str, chunksize=100_000):
    total += len(bloco)
    n = bloco.isna().sum(); nulos = n if nulos is None else nulos + n
    for c in COLS_CODIGO: codigos[c].update(bloco[c].fillna("<nulo>").value_counts().to_dict())
    for c in ["DT_NOTIFIC","DT_SIN_PRI","DT_ENCERRA","DT_DIGITA"]:
        d = pd.to_datetime(bloco[c], errors="coerce")
        datas_min[c] = min(datas_min.get(c, d.min()), d.min()); datas_max[c] = max(datas_max.get(c, d.max()), d.max())
diag_nulos = pd.DataFrame({"nulos": nulos, "nulos_%": (nulos/total*100).round(2)}).sort_values("nulos_%", ascending=False)
print(f"linhas: {total:,} | colunas: {len(nulos)} | {time.time()-t0:.0f} s")
LOG["etapas"]["diagnostico"] = {"linhas": int(total), "colunas": int(len(nulos))}
print("Colunas com mais de 50% de ausência (nulo físico no arquivo):")
print(diag_nulos[diag_nulos["nulos_%"] > 50].to_string())
print("\nColunas sem nenhuma ausência:", diag_nulos[diag_nulos.nulos == 0].index.tolist())
def tabela_codigos(col):
    t = pd.Series(codigos[col]).sort_index(); return pd.DataFrame({"n": t, "%": (t/total*100).round(2)})
for col in ["CLASSI_FIN","CS_SEXO","CS_GESTANT","CS_RACA","CS_ESCOL_N","FEBRE","DIABETES","HOSPITALIZ","EVOLUCAO","CRITERIO","TP_NOT","ID_AGRAVO","NU_ANO"]:
    print(f"\n== {col}"); print(tabela_codigos(col).to_string())
print("Intervalos de datas:")
for c in datas_min: print(f"  {c:12s} {datas_min[c].date()} -> {datas_max[c].date()}")
print("\nCódigos de sintomas e comorbidades efetivamente presentes:")
for c in ["FEBRE","MIALGIA","CEFALEIA","VOMITO","LEUCOPENIA","DIABETES","HIPERTENSA","AUTO_IMUNE"]:
    print(f"  {c:12s}", dict(sorted(codigos[c].items())))
t0 = time.time()
vistos, repetidas = set(), []
with open(ARQ_BRUTO, "rb") as f:
    next(f)                                   # cabeçalho
    for i, linha in enumerate(f):
        h = hashlib.blake2b(linha, digest_size=12).digest()
        if h in vistos: repetidas.append(i)
        else: vistos.add(h)
del vistos
print(f"linhas totalmente duplicadas (todos os 121 campos iguais): {len(repetidas):,} de {total:,} | {time.time()-t0:.0f} s")
LOG["etapas"]["diagnostico"]["duplicatas_exatas"] = len(repetidas)

# ==========================================================================================
# 6. Regras de limpeza
# ==========================================================================================
# ============================================================
# 6. REGRAS DE LIMPEZA
# ============================================================
SINTOMAS = ["FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS","CONJUNTVIT","ARTRITE","ARTRALGIA","PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO"]
COMORB   = ["DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
DEMOG    = ["NU_IDADE_N","CS_SEXO","CS_GESTANT","CS_RACA","CS_ESCOL_N","ID_OCUPA_N"]
LOCAL    = ["SG_UF","ID_MN_RESI"]
DATAS_INI= ["DT_NOTIFIC","DT_SIN_PRI","SEM_NOT","SEM_PRI","NU_ANO"]
ALVO     = ["CLASSI_FIN"]
ALARME   = ["ALRM_HIPOT","ALRM_PLAQ","ALRM_VOM","ALRM_SANG","ALRM_HEMAT","ALRM_ABDOM","ALRM_LETAR","ALRM_HEPAT","ALRM_LIQ","DT_ALRM"]
GRAVID   = ["GRAV_PULSO","GRAV_CONV","GRAV_ENCH","GRAV_INSUF","GRAV_TAQUI","GRAV_EXTRE","GRAV_HIPOT","GRAV_HEMAT","GRAV_MELEN","GRAV_METRO","GRAV_SANG","GRAV_AST","GRAV_MIOC","GRAV_CONSC","GRAV_ORGAO","DT_GRAV"]
DESFECHO = ["EVOLUCAO","DT_OBITO","DT_ENCERRA","HOSPITALIZ","DT_INTERNA","UF","MUNICIPIO","CRITERIO","DOENCA_TRA","CLINC_CHIK"]
LABORAT  = ["DT_CHIK_S1","DT_CHIK_S2","DT_PRNT","RES_CHIKS1","RES_CHIKS2","RESUL_PRNT","DT_SORO","RESUL_SORO","DT_NS1","RESUL_NS1","DT_VIRAL","RESUL_VI_N","DT_PCR","RESUL_PCR_","SOROTIPO","HISTOPA_N","IMUNOH_N"]
INFECCAO = ["TPAUTOCTO","COUFINF","COPAISINF","COMUNINF"]
HEMORRAG = ["MANI_HEMOR","EPISTAXE","GENGIVO","METRO","PETEQUIAS","HEMATURA","SANGRAM","LACO_N","PLASMATICO","EVIDENCIA","PLAQ_MENOR","CON_FHD","COMPLICA"]
ADMIN    = ["TP_NOT","ID_AGRAVO","ID_REGIONA","ID_UNIDADE","ID_RG_RESI","ID_PAIS","DT_INVEST","NDUPLIC_N","DT_DIGITA","CS_FLXRET","FLXRECEBI","MIGRADO_W","TP_SISTEMA"]
OUTRAS_REMOVIDAS = ["SG_UF_NOT","ID_MUNICIP","ANO_NASC"]

decisao = {}
for grupo, cols, destino, motivo in [
    ("demográficas", DEMOG, "retida", "perfil do caso no momento da notificação"),
    ("localização", LOCAL, "retida", "chave territorial e UF de residência"),
    ("sintomas", SINTOMAS, "retida", "quadro clínico inicial da ficha"),
    ("comorbidades", COMORB, "retida", "condições preexistentes da ficha"),
    ("datas iniciais", DATAS_INI, "retida", "tempo de sintomas e notificação; semanas epidemiológicas"),
    ("alvo", ALVO, "alvo", "origem da variável alvo"),
    ("sinais de alarme", ALARME, "auditoria", "critério clínico da classificação 11 (leakage)"),
    ("sinais de gravidade", GRAVID, "auditoria", "critério clínico da classificação 12 (leakage)"),
    ("desfecho e encerramento", DESFECHO, "auditoria", "posterior à notificação ou definido no encerramento (leakage)"),
    ("laboratoriais", LABORAT, "auditoria", "resultados posteriores à notificação"),
    ("local de infecção", INFECCAO, "auditoria", "preenchido na investigação, não na notificação"),
    ("manifestações hemorrágicas (ficha antiga)", HEMORRAG, "auditoria", "campos da ficha anterior, constantes neste arquivo"),
    ("administrativas", ADMIN, "removida", "controle operacional da notificação (item 3.2 do plano)"),
    ("outras", OUTRAS_REMOVIDAS, "removida", "local de notificação não usado; ANO_NASC redundante com NU_IDADE_N")]:
    for c in cols: decisao[c] = {"grupo": grupo, "destino": destino, "motivo": motivo}
todas = list(amostra.columns)
faltam = [c for c in todas if c not in decisao]; sobram = [c for c in decisao if c not in todas]
assert not faltam and not sobram, (faltam, sobram)
tab_decisao = pd.DataFrame.from_dict(decisao, orient="index").loc[todas].rename_axis("coluna").reset_index()
tab_decisao["nulos_%_bruto"] = tab_decisao.coluna.map(diag_nulos["nulos_%"])
tab_decisao.to_csv(ARQ_DECISAO_COLS, index=False)
print(tab_decisao.destino.value_counts().to_string()); print()
print(tab_decisao.groupby("destino").coluna.apply(list).to_string())

# ==========================================================================================
# 7. Aplicação
# ==========================================================================================
# ============================================================
# 7. APLICAÇÃO
# ============================================================
t0 = time.time()
COLS_CARGA = [c for c in todas if decisao[c]["destino"] in ("retida", "alvo")]
del amostra; gc.collect()
COLS_DATA  = [c for c in COLS_CARGA if c.startswith("DT_")]
COLS_TEXTO = ["CS_SEXO","ID_OCUPA_N"]   # texto -> categoria
COLS_NUM   = [c for c in COLS_CARGA if c not in COLS_DATA + COLS_TEXTO]
rep = np.array(sorted(repetidas), dtype=np.int64)
partes, inicio = [], 0
for bloco in pd.read_csv(ARQ_BRUTO, sep=separador, encoding=encoding, usecols=COLS_CARGA, dtype=str, chunksize=200_000):
    bloco.index = np.arange(inicio, inicio + len(bloco)); inicio += len(bloco)
    bloco = bloco.drop(index=rep[(rep >= bloco.index[0]) & (rep <= bloco.index[-1])])
    for c in COLS_NUM:  bloco[c] = pd.to_numeric(bloco[c], errors="coerce").astype("float32")
    for c in COLS_DATA: bloco[c] = pd.to_datetime(bloco[c], errors="coerce")
    partes.append(bloco)
sinan = pd.concat(partes); del partes; gc.collect()
for c in COLS_TEXTO: sinan[c] = sinan[c].astype("category")
sinan.index.name = "id_registro"; sinan = sinan.reset_index()
print(f"carregado: {sinan.shape} | duplicatas exatas removidas: {total - len(sinan):,} | memória {sinan.memory_usage(deep=True).sum()/1e6:.0f} MB | {time.time()-t0:.0f} s")
LOG["etapas"]["carregamento"] = {"linhas_apos_remover_duplicatas_exatas": int(len(sinan)), "colunas_carregadas": len(COLS_CARGA)}
# R3 — alvo, com registro das exclusões antes de excluir
contagem = {"total_original": int(total),
            "duplicatas_exatas_removidas": int(total - len(sinan)),
            "dengue_10": int((sinan.CLASSI_FIN == 10).sum()),
            "dengue_sinais_alarme_11": int((sinan.CLASSI_FIN == 11).sum()),
            "dengue_grave_12": int((sinan.CLASSI_FIN == 12).sum()),
            "inconclusivo_8": int((sinan.CLASSI_FIN == 8).sum()),
            "codigo_0_invalido": int((sinan.CLASSI_FIN == 0).sum()),
            "descartado_5": int((sinan.CLASSI_FIN == 5).sum()),
            "sem_classificacao_nulo": int(sinan.CLASSI_FIN.isna().sum())}
elegivel = sinan.CLASSI_FIN.isin([10, 11, 12])
contagem["excluidos_por_classificacao"] = int((~elegivel).sum())
contagem["base_final_elegivel"] = int(elegivel.sum())
LOG["etapas"]["alvo"] = contagem
print(pd.Series(contagem).to_string())
sinan = sinan.loc[elegivel]; gc.collect()
sinan["alvo"] = (sinan.CLASSI_FIN > 10).astype("int8")
print(f"\nalvo: {int(sinan.alvo.sum()):,} positivos em {len(sinan):,} ({sinan.alvo.mean()*100:.2f}%)")
# R4 — sintomas e comorbidades: só o código 2 (confirmado como "não") vira 0
def recodificar_sim_nao(s, codigos_nulos):
    return s.map({1: 1.0, 2: 0.0}).astype("float32")     # qualquer outro código (9, nulo) permanece nulo
for c in SINTOMAS: sinan[c + "_bin"] = recodificar_sim_nao(sinan[c], [9])
for c in COMORB:   sinan[c + "_bin"] = recodificar_sim_nao(sinan[c], [9])

# R4 — demográficas decodificadas; ignorado e não se aplica como categorias
sinan["sexo"] = sinan.CS_SEXO.astype(str).where(sinan.CS_SEXO.isin(["M","F"]), "I").astype("category")
sinan["gestante"] = np.select([sinan.CS_GESTANT.isin([1,2,3,4]), sinan.CS_GESTANT == 5, sinan.CS_GESTANT == 6], ["sim","nao","nao_se_aplica"], "ignorado")
sinan["gestante"] = sinan.gestante.astype("category")
sinan["raca"] = sinan.CS_RACA.map({1:"branca",2:"preta",3:"amarela",4:"parda",5:"indigena"}).fillna("ignorado").astype("category")
ESC = {0:"sem_instrucao",1:"fund_incompleto",2:"fund_incompleto",3:"fund_incompleto",4:"fund_completo",5:"medio_incompleto",6:"medio_completo",7:"sup_incompleto",8:"sup_completo",10:"nao_se_aplica"}
sinan["escolaridade"] = sinan.CS_ESCOL_N.map(ESC).fillna("ignorado").astype("category")
sinan["ocupacao_cbo"] = sinan.ID_OCUPA_N          # mantida como texto; 62% ausente no bruto

# R5 — idade
pref, val = sinan.NU_IDADE_N // 1000, sinan.NU_IDADE_N % 1000
idade = pd.Series(np.nan, index=sinan.index, dtype="float32")
idade[pref == 4] = val[pref == 4]; idade[pref == 3] = val[pref == 3]/12; idade[pref == 2] = val[pref == 2]/365; idade[pref == 1] = val[pref == 1]/8760
n_idade_invalida = int((idade > 120).sum()); idade[idade > 120] = np.nan
sinan["idade_anos"] = idade

# R6 — datas e intervalo
sinan["dias_ate_notificacao"] = (sinan.DT_NOTIFIC - sinan.DT_SIN_PRI).dt.days.astype("float32")
neg = int((sinan.dias_ate_notificacao < 0).sum()); longo = int((sinan.dias_ate_notificacao > 180).sum())
print(f"dias_ate_notificacao: negativos {neg} | acima de 180 dias {longo} | idades > 120 anos {n_idade_invalida}")
sinan.loc[(sinan.dias_ate_notificacao < 0) | (sinan.dias_ate_notificacao > 180), "dias_ate_notificacao"] = np.nan
sinan["mes_sintomas"] = sinan.DT_SIN_PRI.dt.month.astype("int8")
sinan["competencia"] = (sinan.DT_SIN_PRI.dt.year*100 + sinan.DT_SIN_PRI.dt.month).astype("int32")

# R7 — chave territorial
sinan["cod_municipio"] = sinan.ID_MN_RESI.astype("Int64").astype(str).str.zfill(6).where(sinan.ID_MN_RESI.notna(), None).astype("category")
sinan["uf_res"] = sinan.SG_UF.astype("Int64").astype(str).astype("category")

# R2 — possíveis duplicidades de notificação: marcadas, não removidas
chave = ["DT_NOTIFIC","DT_SIN_PRI","cod_municipio","NU_IDADE_N","CS_SEXO","CS_RACA","CLASSI_FIN"]
sinan["flag_possivel_duplicidade"] = sinan.duplicated(chave, keep=False).astype("int8")
LOG["etapas"]["aplicacao"] = {"idades_invalidas_para_nulo": n_idade_invalida, "dias_negativos_para_nulo": neg, "dias_acima_180_para_nulo": longo,
                              "possiveis_duplicidades_marcadas": int(sinan.flag_possivel_duplicidade.sum())}
print("possíveis duplicidades marcadas (mantidas):", f"{int(sinan.flag_possivel_duplicidade.sum()):,}")

# ==========================================================================================
# 8. Validação
# ==========================================================================================
# ============================================================
# 8. VALIDAÇÃO
# ============================================================
print("R4 — sintoma FEBRE: código original x recodificado"); print(pd.crosstab(sinan.FEBRE.fillna(-1), sinan.FEBRE_bin.fillna(-1), margins=True))
print("\nR4 — comorbidade DIABETES: código original x recodificado (9 -> nulo)"); print(pd.crosstab(sinan.DIABETES.fillna(-1), sinan.DIABETES_bin.fillna(-1), margins=True))
print("\nR4 — CS_GESTANT x gestante"); print(pd.crosstab(sinan.CS_GESTANT.fillna(-1), sinan.gestante))
checks = {
 "id_registro unico": bool(sinan.id_registro.is_unique),
 "alvo so 0/1": set(sinan.alvo.unique()) <= {0, 1},
 "alvo coerente com CLASSI_FIN": bool(((sinan.CLASSI_FIN > 10).astype(int) == sinan.alvo).all()),
 "idade 0-120": bool(sinan.idade_anos.dropna().between(0, 120).all()),
 "dias_ate_notificacao 0-180": bool(sinan.dias_ate_notificacao.dropna().between(0, 180).all()),
 "SEM_PRI em 2025": bool(sinan.SEM_PRI.between(202501, 202553).all()),
 "cod_municipio 6 digitos": bool(sinan.cod_municipio.dropna().str.len().eq(6).all()),
 "sem nulos novos em sintomas (alem de 9/nulo original)": bool((sinan[[c+"_bin" for c in SINTOMAS]].isna().sum().sum() == (~sinan[SINTOMAS].isin([1,2])).sum().sum())),
}
print(pd.Series(checks).to_string()); assert all(checks.values())
print("\nDatas:", sinan.DT_SIN_PRI.min().date(), "->", sinan.DT_SIN_PRI.max().date(), "| notificação até", sinan.DT_NOTIFIC.max().date())
print("cod_municipio nulos:", int(sinan.cod_municipio.isna().sum()), "| municípios distintos:", sinan.cod_municipio.nunique())
LOG["etapas"]["validacao"] = checks
ausencia = (sinan[[c+"_bin" for c in SINTOMAS+COMORB] + ["idade_anos","dias_ate_notificacao","cod_municipio","ocupacao_cbo"]].isna().mean()*100).round(2)
print("Ausência (%) na base tratada, por variável:"); print(ausencia[ausencia > 0].to_string())
print("\nCategorias de ignorado / não se aplica (mantidas):")
for c in ["sexo","gestante","raca","escolaridade"]: print(f"  {c:13s}", (sinan[c].value_counts(normalize=True)*100).round(2).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(sinan.idade_anos.dropna(), bins=60, color="steelblue"); ax[0].set_title("Idade em anos após decodificação"); ax[0].set_xlabel("anos")
ax[1].hist(sinan.dias_ate_notificacao.dropna().clip(upper=30), bins=31, color="darkorange"); ax[1].set_title("Dias até a notificação (truncado em 30)"); ax[1].set_xlabel("dias")
plt.tight_layout(); plt.show()

# ==========================================================================================
# 9. Exportação
# ==========================================================================================
# ============================================================
# 9. EXPORTAÇÃO
# ============================================================
COLS_SAIDA = (["id_registro","DT_NOTIFIC","DT_SIN_PRI","SEM_NOT","SEM_PRI","NU_ANO","NU_IDADE_N","idade_anos",
               "CS_SEXO","sexo","CS_GESTANT","gestante","CS_RACA","raca","CS_ESCOL_N","escolaridade","ocupacao_cbo",
               "SG_UF","uf_res","ID_MN_RESI","cod_municipio"] + [c+"_bin" for c in SINTOMAS+COMORB]
              + ["dias_ate_notificacao","mes_sintomas","competencia","CLASSI_FIN","alvo","flag_possivel_duplicidade"])
COLS_AUDIT = ["id_registro","CLASSI_FIN","alvo"] + [c for c in todas if decisao[c]["destino"] == "auditoria"]

tratado = sinan[COLS_SAIDA]
tratado.to_csv(ARQ_TRATADO, index=False, date_format="%Y-%m-%d")
tratado.to_parquet(ARQ_TRATADO.with_suffix(".parquet"), index=False)
# auditoria gravada em fluxo: mesmas linhas (sem duplicatas exatas, só elegíveis), sem ocupar memória
t0 = time.time(); inicio = 0; primeiro = True
for bloco in pd.read_csv(ARQ_BRUTO, sep=separador, encoding=encoding, usecols=[c for c in COLS_AUDIT if c not in ("id_registro","alvo")], dtype=str, chunksize=200_000):
    bloco.index = np.arange(inicio, inicio + len(bloco)); inicio += len(bloco)
    bloco = bloco.drop(index=rep[(rep >= bloco.index[0]) & (rep <= bloco.index[-1])])
    bloco = bloco[bloco.CLASSI_FIN.isin(["10","11","12"])]
    bloco.insert(0, "id_registro", bloco.index); bloco.insert(2, "alvo", (bloco.CLASSI_FIN != "10").astype(int))
    bloco.to_csv(ARQ_AUDITORIA, index=False, mode="w" if primeiro else "a", header=primeiro); primeiro = False
print(f"auditoria gravada em {time.time()-t0:.0f} s")

regra = {"id_registro": "posição da linha no arquivo bruto (0 = primeira linha de dados)",
         "idade_anos": "R5: NU_IDADE_N decodificada (4 anos, 3 meses, 2 dias, 1 horas); > 120 -> nulo",
         "sexo": "R4: CS_SEXO M/F; demais -> I", "gestante": "R4: CS_GESTANT 1-4 sim, 5 nao, 6 nao_se_aplica, demais ignorado",
         "raca": "R4: CS_RACA 1 branca, 2 preta, 3 amarela, 4 parda, 5 indigena, demais ignorado",
         "escolaridade": "R4: CS_ESCOL_N decodificada; 9/nulo ignorado; 10 nao_se_aplica", "ocupacao_cbo": "ID_OCUPA_N sem alteração (62% ausente)",
         "uf_res": "SG_UF como texto", "cod_municipio": "R7: ID_MN_RESI com 6 dígitos, texto",
         "dias_ate_notificacao": "R6: DT_NOTIFIC - DT_SIN_PRI em dias; <0 ou >180 -> nulo", "mes_sintomas": "R6: mês de DT_SIN_PRI",
         "competencia": "R6: AAAAMM de DT_SIN_PRI (chave temporal do CNES)", "alvo": "R3: CLASSI_FIN 10 -> 0; 11,12 -> 1",
         "flag_possivel_duplicidade": "R2: 1 se a chave (datas, município, idade, sexo, raça, classificação) se repete; mantida"}
for c in SINTOMAS: regra[c+"_bin"] = f"R4: {c} 1 -> 1, 2 -> 0, demais -> nulo"
for c in COMORB:   regra[c+"_bin"] = f"R4: {c} 1 -> 1, 2 -> 0, 9/nulo -> nulo"
dicio = pd.DataFrame({"coluna": COLS_SAIDA, "dtype": tratado.dtypes.astype(str).values,
                      "origem_regra": [regra.get(c, "coluna original do SINAN, sem alteração") for c in COLS_SAIDA]})
dicio.to_csv(ARQ_DICIONARIO, index=False)
LOG["etapas"]["exportacao"] = {"linhas": int(len(tratado)), "colunas": len(COLS_SAIDA), "tamanho_MB": round(ARQ_TRATADO.stat().st_size/1e6, 1),
                               "auditoria_colunas": len(COLS_AUDIT)}
aud_ids = pd.read_csv(ARQ_AUDITORIA, usecols=["id_registro"]).id_registro
assert len(aud_ids) == len(tratado) and (aud_ids.values == tratado.id_registro.values).all(), "auditoria e base tratada divergem"
print("auditoria alinhada à base tratada:", len(aud_ids), "linhas")
print(json.dumps(LOG["etapas"]["exportacao"], indent=2))
print(dicio)

# ==========================================================================================
# 10. Resumo
# ==========================================================================================
# ============================================================
# 10. RESUMO
# ============================================================
LOG["executado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
(ARQ_LOG).write_text(json.dumps(LOG, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
resumo = pd.DataFrame([
    ["linhas no bruto", total, ""],
    ["linhas totalmente duplicadas removidas (R2)", contagem["duplicatas_exatas_removidas"], "hash da linha completa"],
    ["excluídas por classificação (R3)", contagem["excluidos_por_classificacao"], "8 inconclusivo, 0 inválido, nulo"],
    ["base final elegível", contagem["base_final_elegivel"], "CLASSI_FIN 10, 11, 12"],
    ["positivos (alvo = 1)", int(sinan.alvo.sum()), f"{sinan.alvo.mean()*100:.2f}%"],
    ["colunas no bruto", len(todas), ""],
    ["colunas retidas / alvo / auditoria / removidas", f"{(tab_decisao.destino=='retida').sum()} / 1 / {(tab_decisao.destino=='auditoria').sum()} / {(tab_decisao.destino=='removida').sum()}", "R1"],
    ["colunas na base tratada", len(COLS_SAIDA), "originais + derivadas"],
    ["possíveis duplicidades marcadas", LOG["etapas"]["aplicacao"]["possiveis_duplicidades_marcadas"], "mantidas com flag"],
    ["idades inválidas -> nulo", n_idade_invalida, "R5"],
    ["dias até notificação inválidos -> nulo", neg + longo, "R6"],
], columns=["item", "valor", "observação"])
print(resumo)
