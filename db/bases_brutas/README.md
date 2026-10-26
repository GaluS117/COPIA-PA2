# db/bases_brutas — fontes originais (imutáveis)

Nenhuma rotina escreve nesta pasta. A identidade de cada arquivo (tamanho, SHA-256) é registrada pelos notebooks de limpeza
nos logs de `db/bases_tratadas/<fonte>/*_limpeza_log.json`.

| Pasta | Arquivo(s) | Origem |
|---|---|---|
| `sinan/dados/` | `DENGBR25.csv` | DATASUS – SINAN, dengue (A90), notificações com primeiros sintomas em 2025 |
| `sinan/dicionario/` | dicionário da ficha de investigação de dengue (PDF/XLSX) | DATASUS |
| `cnes/` | `Leitos_2025.csv` | DATASUS – CNES, leitos por estabelecimento e competência |
| `ibge/` | `Agregados_por_municipios_basico_BR.csv`, `..._caracteristicas_domicilio{1,2,3}_BR.csv` | IBGE – Censo 2022, agregados por municípios |
| `ibge/dicionario/` | dicionário XLSX do IBGE + `mapeamento_indicadores.csv` (ver README da pasta) | IBGE / grupo |
| `informes/` | 17 informes semanais de arboviroses de 2025 (PDF) | Ministério da Saúde |

Versionados via Git LFS (ver `.gitattributes`).
