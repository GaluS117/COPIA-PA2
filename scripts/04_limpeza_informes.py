#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execução reproduzível de notebooks/04_limpeza_informes.ipynb (gerado a partir das células de código).
Uso, a partir da raiz do repositório:  python scripts/04_limpeza_informes.py"""
import matplotlib; matplotlib.use("Agg")   # sem janela gráfica
def display(x): print(x)


# ==========================================================================================
# 1. Objetivo
# ==========================================================================================
# ============================================================
# 1. OBJETIVO
# ============================================================
ENTRADA_PREVISTA = "db/bases_brutas/informes/*.pdf"
SAIDA_PREVISTA   = "db/bases_tratadas/informes/informes_corpus_2025.csv"

# ==========================================================================================
# 2. Bibliotecas
# ==========================================================================================
# ============================================================
# 2. BIBLIOTECAS
# ============================================================
import re, json, hashlib, unicodedata, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pdfplumber
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_colwidth", 120)
print("pdfplumber", pdfplumber.__version__, "| pandas", pd.__version__)

# ==========================================================================================
# 3. Caminhos
# ==========================================================================================
# ============================================================
# 3. CAMINHOS
# ============================================================
RAIZ = Path(__file__).resolve().parents[1]
BRUTA, TRATADA = RAIZ/"db/bases_brutas/informes", RAIZ/"db/bases_tratadas/informes"
TRATADA.mkdir(parents=True, exist_ok=True)
ARQ_CORPUS   = TRATADA/"informes_corpus_2025.csv"
ARQ_CABEC    = TRATADA/"informes_cabecalhos_rodapes_removidos.csv"
ARQ_SECOES   = TRATADA/"informes_secoes_detectadas.csv"
ARQ_LOG      = TRATADA/"informes_limpeza_log.json"
PDFS = sorted(BRUTA.glob("*.pdf"))
LOG = {"entrada": str(BRUTA.relative_to(RAIZ)), "saida": str(ARQ_CORPUS.relative_to(RAIZ)), "n_informes": len(PDFS), "etapas": {}}
print(f"{len(PDFS)} informe(s) encontrado(s) em {BRUTA.relative_to(RAIZ)}")
for p in PDFS: print("  ", p.name)
if not PDFS: print("\n>>> Coloque os 17 PDFs em db/bases_brutas/informes/ e execute novamente. As seções seguintes rodam sem erro e não gravam corpus vazio.")

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

def carregar_pdf(caminho):
    paginas = []
    with pdfplumber.open(caminho) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            altura = float(pg.height) or 1.0
            try:
                linhas = [{"texto": l["text"], "top": l["top"]/altura,
                           "size": max(c["size"] for c in l["chars"]), "bold": any("Bold" in c["fontname"] for c in l["chars"])}
                          for l in pg.extract_text_lines(strip=True, return_chars=True)]
            except Exception:
                linhas = [{"texto": t, "top": np.nan, "size": np.nan, "bold": False} for t in (pg.extract_text() or "").splitlines()]
            paginas.append({"pagina": i, "texto_original": pg.extract_text() or "", "linhas": linhas, "n_chars": len(pg.chars)})
    return paginas

RE_NUM_INFORME = re.compile(r"n[º°o\.]*\s*(\d{1,2})", re.I)
RE_SE = re.compile(r"SE\s*(\d{1,2})\s*a\s*(?:SE\s*)?(\d{1,2})\s*/\s*(20\d\d)", re.I)
RE_DATA = re.compile(r"\((\d{2}/\d{2}/\d{4})\)")
def metadados(nome, pags):
    texto_ini = " ".join(p["texto_original"] for p in pags[:2])
    num = RE_NUM_INFORME.search(nome); se = RE_SE.search(texto_ini); dt = RE_DATA.search(texto_ini)
    return {"informe_num": int(num.group(1)) if num else None, "se_inicio": int(se.group(1)) if se else None, "se_fim": int(se.group(2)) if se else None,
            "ano_se": int(se.group(3)) if se else None, "data_informe": pd.to_datetime(dt.group(1), dayfirst=True).date().isoformat() if dt else None}

t0 = time.time(); docs = {}
meta = {}
for p in PDFS:
    docs[p.name] = carregar_pdf(p); meta[p.name] = metadados(p.name, docs[p.name])
    LOG["etapas"].setdefault("identidade", {})[p.name] = {"tamanho_KB": round(p.stat().st_size/1e3), "sha256": sha256(p), "paginas": len(docs[p.name]), **meta[p.name]}
print(f"{len(docs)} documento(s), {sum(len(v) for v in docs.values())} páginas, {time.time()-t0:.0f} s")
meta = pd.DataFrame(meta).T.sort_values("informe_num") if meta else pd.DataFrame()
print(meta)

# ==========================================================================================
# 5. Diagnóstico inicial
# ==========================================================================================
# ============================================================
# 5. DIAGNÓSTICO INICIAL
# ============================================================
def normalizar_chave(t):   # chave para comparar linhas repetidas: sem acento, minúsculas, sem dígitos (números de página/semana variam)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", t.lower())).strip()

diag = []
repetidas = {}
for nome, pags in docs.items():
    n = len(pags); sem_texto = sum(1 for p in pags if p["n_chars"] == 0)
    chars = [len(p["texto_original"]) for p in pags]
    cont = Counter(normalizar_chave(l["texto"]) for p in pags for l in p["linhas"] if l["texto"].strip())
    rep = {k: v for k, v in cont.items() if n >= 3 and v >= max(3, 0.5*n)}
    repetidas[nome] = rep
    diag.append({"informe": nome, "paginas": n, "paginas_sem_texto": sem_texto, "chars_medio_pagina": int(np.mean(chars)) if chars else 0,
                 "chars_min": min(chars) if chars else 0, "linhas_repetidas_50%": len(rep)})
diag = pd.DataFrame(diag)
print(diag.to_string() if len(diag) else "sem documentos")
for nome, rep in list(repetidas.items())[:3]:
    print(f"\n== {nome}: linhas repetidas em ≥50% das páginas (candidatas a cabeçalho/rodapé)")
    for k, v in sorted(rep.items(), key=lambda kv: -kv[1])[:8]: print(f"   [{v:>3} págs] {k[:100]}")
# tamanhos de fonte por documento: o corpo dos informes está em 14 pt e os títulos de seção em negrito ≥ 16 pt
fontes = Counter()
for nome, pags in docs.items():
    for p in pags:
        for l in p["linhas"]:
            if l["size"] == l["size"]: fontes[(round(l["size"]), l["bold"])] += 1
print("combinações (tamanho, negrito) mais frequentes nas linhas:", fontes.most_common(8))
titulos = Counter()
for nome, pags in docs.items():
    for p in pags:
        for l in p["linhas"]:
            if l["bold"] and l["size"] >= 15.5 and len(l["texto"].split()) <= 8 and not re.search(r"\d", l["texto"]): titulos[l["texto"].strip()] += 1
print("\ntítulos em negrito ≥ 16 pt (frequência nos", len(docs), "informes):")
for t, n in titulos.most_common(20): print(f"  {n:3d} | {t}")

# ==========================================================================================
# 6. Regras de limpeza
# ==========================================================================================
# ============================================================
# 6. REGRAS DE LIMPEZA
# ============================================================
TOPO, BASE = 0.12, 0.88
RE_NUM_PAG = re.compile(r"^\s*(p[áa]g(ina)?\.?\s*)?\d{1,3}(\s*(/|de)\s*\d{1,3})?\s*$", re.I)
RE_INICIO_DENGUE = re.compile(r"\bdengue\b", re.I)
RE_FIM_DENGUE    = re.compile(r"\b(chikungunya|zika|febre amarela|oropouche|mayaro|introdu[cç][aã]o|a[cç][oõ]es realizadas|insumos|vacina[cç][aã]o|refer[eê]ncias)\b", re.I)
RE_TITULO = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^.]{2,80}$")

def e_cabecalho_rodape(linha, rep_doc):
    t = linha["texto"].strip(); top = linha["top"]
    if not t: return True
    borda = (top != top) or top < TOPO or top > BASE                       # sem posição, ou no topo/base da página
    if RE_NUM_PAG.match(t): return borda                                   # número solto só é nº de página se estiver na borda
    return normalizar_chave(t) in rep_doc and borda

def e_titulo(linha):
    t = linha["texto"].strip(); n = len(t.split())
    if linha["size"] == linha["size"]:                                   # há informação de fonte
        secao  = linha["bold"] and linha["size"] >= 15.5 and n <= 8
        tabela = linha["bold"] and 11.5 <= linha["size"] < 15.5 and linha["top"] <= TOPO and n <= 20
        return secao or tabela
    return bool(RE_TITULO.match(t)) and n <= 12                          # sem fonte: aparência da linha

def limpar_texto(t):
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)            # hifenização na quebra de linha
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ==========================================================================================
# 7. Aplicação
# ==========================================================================================
# ============================================================
# 7. APLICAÇÃO
# ============================================================
corpus, removidas, secoes = [], [], []
for nome, pags in docs.items():
    rep_doc = repetidas.get(nome, {}); em_dengue = False; tipo_atual = "narrativa"
    for p in pags:
        mantidas, dengue_linhas = [], []; tipos_pag = set()
        for l in p["linhas"]:
            t = l["texto"].strip()
            if e_cabecalho_rodape(l, rep_doc):
                if t: removidas.append({"informe": nome, "pagina": p["pagina"], "linha": t}); 
                continue
            if e_titulo(l):
                if RE_INICIO_DENGUE.search(t):
                    em_dengue = True; tipo_atual = "tabela" if (l["size"] == l["size"] and l["size"] < 15.5) else "narrativa"
                    secoes.append({"informe": nome, "pagina": p["pagina"], "titulo": t, "evento": "inicio_dengue", "tipo": tipo_atual})
                elif em_dengue and RE_FIM_DENGUE.search(t): em_dengue = False; secoes.append({"informe": nome, "pagina": p["pagina"], "titulo": t, "evento": "fim_dengue", "tipo": tipo_atual})
            mantidas.append(t)
            if em_dengue: dengue_linhas.append(t); tipos_pag.add(tipo_atual)
        corpus.append({"informe": nome, **(meta.loc[nome].to_dict() if len(meta) else {}), "pagina": p["pagina"], "sem_texto": int(p["n_chars"] == 0), "n_caracteres": len(p["texto_original"]),
                       "texto_original": p["texto_original"], "texto_limpo": limpar_texto("\n".join(mantidas)),
                       "secao_dengue": int(bool(dengue_linhas)), "tipo_secao_dengue": "/".join(sorted(tipos_pag)) if tipos_pag else "",
                       "texto_dengue": limpar_texto("\n".join(dengue_linhas))})
corpus, removidas, secoes = pd.DataFrame(corpus), pd.DataFrame(removidas), pd.DataFrame(secoes)
if len(corpus): corpus = corpus.sort_values(["informe_num","pagina"]).reset_index(drop=True)
print(f"corpus: {len(corpus)} páginas | linhas de cabeçalho/rodapé removidas: {len(removidas)} | títulos de seção detectados: {len(secoes)}")
LOG["etapas"]["aplicacao"] = {"paginas": int(len(corpus)), "linhas_removidas": int(len(removidas)), "titulos_secao": int(len(secoes)),
                              "paginas_com_secao_dengue": int(corpus.secao_dengue.sum()) if len(corpus) else 0}

# ==========================================================================================
# 8. Validação
# ==========================================================================================
# ============================================================
# 8. VALIDAÇÃO
# ============================================================
if len(corpus):
    checks = {
     "informe e pagina preenchidos": bool(corpus[["informe","pagina"]].notna().all().all()),
     "(informe, pagina) unico": bool(not corpus.duplicated(["informe","pagina"]).any()),
     "texto_limpo <= texto_original": bool((corpus.texto_limpo.str.len() <= corpus.texto_original.str.len() + 5).all()),
     "secao_dengue so em informes com titulo de dengue": bool(set(corpus.loc[corpus.secao_dengue == 1, "informe"]) <= set(secoes.loc[secoes.evento == "inicio_dengue", "informe"]) if len(secoes) else corpus.secao_dengue.sum() == 0),
     "paginas sem texto marcadas": bool(((corpus.n_caracteres == 0) == (corpus.sem_texto == 1)).all()),
    }
    print(pd.Series(checks).to_string()); assert all(checks.values()); LOG["etapas"]["validacao"] = checks
    print("\npáginas por informe com seção de dengue:"); print(corpus.groupby("informe_num").agg(se=("se_fim","first"), paginas=("pagina","size"), com_dengue=("secao_dengue","sum"), narrativa=("tipo_secao_dengue", lambda s: (s == "narrativa").sum()), tabela=("tipo_secao_dengue", lambda s: (s == "tabela").sum())).to_string())
    print("\namostra de linhas removidas:"); print(removidas.sample(min(10, len(removidas)), random_state=42).to_string() if len(removidas) else "nenhuma")
    print("\ninício das seções de dengue detectadas:"); print(secoes[secoes.evento == "inicio_dengue"].to_string() if len(secoes) else "nenhuma")
else:
    print("nada a validar: nenhum PDF na camada bruta")

# ==========================================================================================
# 9. Exportação
# ==========================================================================================
# ============================================================
# 9. EXPORTAÇÃO
# ============================================================
if len(corpus):
    corpus.to_csv(ARQ_CORPUS, index=False); removidas.to_csv(ARQ_CABEC, index=False); secoes.to_csv(ARQ_SECOES, index=False)
    LOG["etapas"]["exportacao"] = {"linhas": int(len(corpus)), "colunas": list(corpus.columns), "tamanho_KB": round(ARQ_CORPUS.stat().st_size/1e3)}
    print(json.dumps(LOG["etapas"]["exportacao"], indent=2, ensure_ascii=False))
else:
    LOG["etapas"]["exportacao"] = "não executada: nenhum PDF"; print("nenhum corpus gravado")
LOG["executado_em"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
ARQ_LOG.write_text(json.dumps(LOG, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# ==========================================================================================
# 10. Resumo
# ==========================================================================================
# ============================================================
# 10. RESUMO
# ============================================================
pd.DataFrame([
    ["informes (PDF) na camada bruta", len(PDFS), "esperados: 17"],
    ["páginas no corpus", len(corpus), "uma linha por página"],
    ["páginas sem camada de texto", int(corpus.sem_texto.sum()) if len(corpus) else 0, "exigiriam OCR"],
    ["linhas removidas (cabeçalho/rodapé/nº página)", len(removidas), "R2, gravadas para conferência"],
    ["títulos de seção detectados", len(secoes), "R4, revisão manual"],
    ["páginas com seção de dengue", int(corpus.secao_dengue.sum()) if len(corpus) else 0, "R4"],
], columns=["item","valor","observação"])
