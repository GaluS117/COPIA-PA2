# Projeto Aplicado II — Análise da dengue com sinais de alarme ou grave (SINAN 2025)

Guilherme Cunha Oliveira (RA 10754621) e Gabriel Lucca Simionato (RA 10751300) — Universidade Presbiteriana Mackenzie (EaD).

Objetivo (Mapeamento de Processos, Etapa 1): identificar perfis associados à classificação de dengue com sinais de
alarme ou dengue grave (`CLASSI_FIN` 11/12 vs 10), usando atributos dos casos (SINAN) e contexto municipal
(CNES leitos e IBGE Censo 2022), com a trilha textual dos Informes Semanais de Arboviroses como complemento.
A limpeza segue o **Plano de Limpeza das Bases** e as demais atividades o **Roteiro Complementar da Entrega 2** (ambos em `docs/`): brutos imutáveis, camada tratada separada, um notebook por fonte; depois EDA, preparação, treinamento, métricas e análise textual.

## Estrutura do repositório

```
db/
├── bases_brutas/        fontes originais, nunca sobrescritas (Git LFS)
│   ├── sinan/dados/  sinan/dicionario/
│   ├── cnes/
│   ├── ibge/  ibge/dicionario/       (mapeamento_indicadores.csv: água, esgoto, lixo — a aprovar)
│   └── informes/                     17 PDFs dos Informes Semanais do COE (nº 01–17, SE 1–4 a SE 1–23)
└── bases_tratadas/      saídas das limpezas + base analítica
    ├── sinan/     sinan_dengue_2025_tratado.csv (+ .parquet), _auditoria.csv, _dicionario.csv, colunas_decisao.csv, log
    ├── cnes/      cnes_municipio_2025_tratado.csv, cnes_estabelecimentos_2025_validado.csv, log
    ├── ibge/      ibge_municipio_2022_tratado.csv, log
    ├── informes/  informes_corpus_2025.csv, cabeçalhos removidos, seções detectadas, log
    └── analitica/ split_ids.csv (teste reservado), base_analitica_dev.parquet, métricas, resumo
notebooks/
├── 01_limpeza_sinan.ipynb          10 seções do plano: objetivo → resumo
├── 02_limpeza_cnes.ipynb
├── 03_limpeza_ibge.ipynb
├── 04_limpeza_informes.ipynb
├── 05_integracao_eda_modelo.ipynb  P03–P05 e P07: reserva do teste, EDA (7 itens do roteiro), preparação, baseline, métricas
└── 06_analise_informes.ipynb       P06: termos, localidades, sorotipos, tendências, TF-IDF, amostra para revisão manual
scripts/          um .py por notebook (gerados das células de código) + executar_tudo.sh
reports/          relatório técnico de cada etapa e figuras
diario/           diário de andamento
docs/             mapeamento de processos, plano de limpeza, materiais de apoio
```

## Como reproduzir

```bash
pip install -r requirements.txt
git lfs pull                      # traz os arquivos brutos reais
bash scripts/executar_tudo.sh     # SINAN (~2 min, 1,5 GB RAM) -> CNES -> IBGE -> Informes -> integração/modelo (~1,5 min) -> análise textual
# ou execute os notebooks na ordem 01 -> 02 -> 03 -> 04 -> 05 -> 06
```

Padrão de processo em todas as fontes: **entrada** (base bruta preservada) → **processamento** (regras documentadas na seção 6 de cada
notebook) → **validação** (seção 8) → **saída** (camada tratada, com log). Semente fixa `SEED = 42`; o conjunto de teste (20%,
estratificado) é reservado em `db/bases_tratadas/analitica/split_ids.csv` antes de qualquer exploração e só será aberto em P08.

## Bibliotecas (item a da Etapa 2)

pandas e NumPy (bases tabulares) · Matplotlib (gráficos) · scikit-learn (pré-processamento, separação, regressão logística, métricas, TF-IDF) · pdfplumber (informes) · openpyxl (dicionário do IBGE) · pyarrow (parquet) · biblioteca padrão (pathlib, hashlib, json, re).
Versões em `requirements.txt`.

## Exigências da Etapa 2 e onde estão atendidas

| Exigência | Evidência |
|---|---|
| Bibliotecas Python e GitHub | `requirements.txt`, este README, seção 2 do relatório |
| Análise exploratória | notebook 05 (Etapas 4–12) e `db/bases_tratadas/analitica/` |
| Tratamento / preparação e treinamento | notebooks 01–04 (limpeza) e 05 (Etapa 13: preparação e ajuste inicial) |
| Bases teóricas do método | seção 5 do relatório |
| Cálculo da acurácia | notebook 05 (Etapa 14) e seção 6 do relatório |

## Estado das etapas

| Processo do mapa | Situação |
|---|---|
| P01 Delimitar a proposta | concluído (Etapa 1) |
| P02 Ler e compreender as fontes | concluído — notebooks 01 a 04 |
| P03 Casos elegíveis e reserva do teste | concluído — 01 (alvo) e 05 (split) |
| P04 Explorar os dados de desenvolvimento | concluído — 05 |
| P05 Preparar e integrar | concluído para SINAN + CNES + IBGE básico; saneamento aguarda `mapeamento_indicadores.csv` aprovado |
| P06 Trilha textual (informes) | concluído — corpus, menções, TF-IDF e indicadores citados dos 17 informes; revisão manual da amostra pendente |
| P07 Ajustar e comparar na validação | baseline concluído (SINAN isolado vs enriquecido) — 05 |
| P08 Avaliar no teste e comunicar | Etapa 3 |
