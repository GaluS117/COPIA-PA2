# Dicionário do IBGE (Censo 2022 – agregados por municípios)

Coloque aqui o dicionário oficial em XLSX baixado com os agregados (ex.: `dicionario_de_dados_agregados_por_municipios.xlsx`).

Preencha `mapeamento_indicadores.csv` a partir dele, uma linha por indicador:

| coluna | conteúdo |
|---|---|
| indicador | nome da coluna de saída (ex.: `pct_agua_rede`) |
| arquivo | `basico`, `domicilio1`, `domicilio2` ou `domicilio3` |
| numerador | código da variável (ex.: `V00123`) confirmado no dicionário |
| denominador | código da variável do total correspondente (ex.: total de DPPO do mesmo arquivo) |
| descricao | texto do dicionário para o numerador |
| aprovado | `sim` para entrar na base tratada; `nao` mantém a linha só como rascunho |

O notebook `03_limpeza_ibge.ipynb` lê apenas as linhas com `aprovado = sim`, confere se os códigos existem
no arquivo indicado e calcula `indicador = numerador / denominador * 100`. Nenhum código é escolhido pelo nome do arquivo.
