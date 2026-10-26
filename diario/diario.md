# Diário do projeto

## 2026-09-25 — Etapa 2 (Guilherme): limpeza por fonte conforme o Plano de Limpeza das Bases

**Estrutura.** Repositório reorganizado em `db/bases_brutas/` (imutável) e `db/bases_tratadas/`, com um notebook por fonte nas 10 seções do plano
(01 SINAN, 02 CNES, 03 IBGE, 04 Informes) e um notebook de integração/EDA/modelo (05) que lê só a camada tratada.

**SINAN (01).** 1.643.215 linhas × 121 colunas. Linhas totalmente duplicadas (hash dos 121 campos): 620, removidas. Decisão por coluna gravada em
`sinan_colunas_decisao.csv`: 34 retidas, 1 alvo, 70 auditoria (leakage), 16 removidas. Alvo: 10 → 0; 11/12 → 1; excluídos 157.976 inconclusivos (8),
1.642 código 0 e 1.175 nulos; não há descartados (5) no arquivo. Base elegível: 1.481.802 casos, 37.792 positivos (2,55%).
Recodificação só do código 2 para "não" (9 e nulo permanecem nulos). 36 idades > 120 anos e 1.937 atrasos > 180 dias viraram nulo.
72.659 possíveis duplicidades por chave de caso marcadas com flag, não removidas.

**CNES (02).** 86.147 linhas, 12 competências, 7.369 estabelecimentos, 3.577 municípios. Sem nulos, negativos ou inconsistências entre categorias.
Agregado para 42.552 linhas município × competência (`leitos`, `leitos_sus`, `leitos_uti`, `leitos_uti_sus`, `n_estabelecimentos`).

**IBGE (03).** 5.570 municípios; usadas v0001 (população), AREA_KM2, v0007 (domicílios ocupados), v0005 (média de moradores); densidade derivada.
Chave de 6 dígitos validada contra o SINAN tratado: 1.481.793 casos com município (9 sem: UU0000 e códigos inexistentes; 5 nulos).
Indicadores de água/esgoto/lixo dependem de `db/bases_brutas/ibge/dicionario/mapeamento_indicadores.csv` aprovado (linhas com `aprovado = sim`).

**Informes (04).** Notebook pronto e testado com outros PDFs (extração por página, remoção de cabeçalho/rodapé por repetição e posição, detecção de
títulos de seção para revisão manual, normalização). Pasta `db/bases_brutas/informes/` ainda vazia.

**Integração e baseline (05).** Teste de 20% reservado (296.361 casos). Junção preservou os 1.185.441 casos de desenvolvimento; 8 sem IBGE; 7,3% em
município sem leitos (zero real). Regressão logística balanceada na validação: AUC-ROC 0,721 (SINAN) vs 0,736 (SINAN + CNES + IBGE).

**Pendências.**
1. Dicionário XLSX do IBGE → preencher e aprovar `mapeamento_indicadores.csv`.
2. 17 PDFs dos informes → `db/bases_brutas/informes/` e executar o notebook 04.
3. Dicionário da ficha SINAN em `db/bases_brutas/sinan/dicionario/` para conferência dos códigos usados.
4. Registrar no README o endereço do repositório e os links de download de cada arquivo.

## 2026-09-26 — Etapa 2 (Guilherme): alinhamento ao Roteiro Complementar da Entrega 2

- Bibliotecas restritas às do roteiro e do mapa (seaborn removido; gráficos só com Matplotlib).
- EDA do notebook 05 cobre explicitamente os sete itens da seção 2 do roteiro: registros antes/depois da elegibilidade, distribuição de CLASSI_FIN e do alvo,
  ausente × ignorado × não se aplica por variável candidata (`eda_ausencias_variaveis_candidatas.csv`), distribuições, consistência de datas e códigos de município,
  indicadores CNES/IBGE após integração e desbalanceamento.
- Preparação para treinamento conforme a tabela da seção 3: imputação por moda com indicador nos 0/1 (ausente não vira "não" sem marca), mediana com indicador nas contínuas,
  padronização só das contínuas, tudo aprendido no treino. Métricas na validação praticamente iguais (AUC-ROC 0,721 / 0,736).
- Método de avaliação definido antes do treino com as cinco métricas da seção 5 (matriz de confusão, acurácia, precisão, recall, F1); acurácia balanceada e AUCs só como apoio;
  critério de honestidade registrado (sem desempenho mínimo, sem promessa de utilidade clínica).
- Notebook 06 de análise textual (seção 6): frequência de termos, expressões (sorotipos, tendência, gravidade, vigilância), localidades por correspondência com o IBGE tratado,
  TF-IDF por informe e amostra de 30 menções para revisão manual. Testado com PDFs de apoio; aguarda os 17 informes.

## 2026-09-26 — Informes Semanais incluídos e trilha textual executada (Guilherme)

- 17 PDFs (COE Dengue e outras Arboviroses, nº 01 a 17, 343 páginas) em `db/bases_brutas/informes/`. Todos com camada de texto.
- Notebook 04: metadados de período lidos do conteúdo (nº, SE inicial/final, data) — o arquivo do nº 08 diz "SE 1 a 13" mas o conteúdo é SE 1 a 14.
  Regra de seção passou a usar a tipografia (corpo 14 pt; títulos de doença em negrito 20 pt; títulos de tabela anexa em negrito 12 pt no alto da página).
  Resultado estável: 120 páginas de dengue (69 narrativa/infográfico, 51 tabelas anexas), 993 linhas de cabeçalho/rodapé removidas, 138 títulos gravados para revisão.
  Correção: número solto só é nº de página se estiver na borda (antes apagava valores do infográfico).
- Notebook 06: termos/TF-IDF só na narrativa (tabelas repetem as 27 UFs); UF/região têm prioridade sobre município homônimo (Mato Grosso/PB, Tocantins/MG…);
  indicadores citados extraídos por informe (casos prováveis, incidência, variação vs 2024, óbitos, sorotipos, tendência nowcasting), cada um no seu período, sem soma.
  Trajetória relatada: 139 mil casos prováveis (SE 4) → 1,48 milhão (SE 23); óbitos 21 → 1.165; DENV-2 predominante a partir do nº 09; tendência de alta até o nº 08, queda/redução depois.
- Relatório: seção 4.5 reescrita com os resultados; tabela dos 17 informes e Figura 5.

**Pendências.** Revisão manual da amostra de 30 menções (`informes_amostra_revisao.csv`); dicionário XLSX do IBGE; endereço do GitHub.
