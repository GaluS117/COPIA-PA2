#!/usr/bin/env bash
# Reproduz limpeza, integração/modelo e análise textual a partir das bases brutas, na ordem do plano e do roteiro complementar.
set -e
cd "$(dirname "$0")/.."
python scripts/01_limpeza_sinan.py          # 1. limpeza das fontes
python scripts/02_limpeza_cnes.py
python scripts/03_limpeza_ibge.py
python scripts/04_limpeza_informes.py
python scripts/05_integracao_eda_modelo.py  # 2-5. EDA, preparação, treinamento, métricas
python scripts/06_analise_informes.py       # 6. análise textual
echo "concluído: db/bases_tratadas/ atualizada"
