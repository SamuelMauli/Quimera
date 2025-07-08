# Projeto Quimera 2.0: O Desafio Tabula Rasa

**Uma IA de Xadrez que Aprende as Regras do Zero para Superar um Mestre**

## Visão Geral

O Projeto Quimera 2.0 representa uma evolução fundamental do conceito original. Mantendo a arquitetura de dupla capacidade (maestria no jogo e modelagem preditiva do oponente), esta versão adota uma abordagem de ***tabula rasa* (mente em branco) radical**.

A IA não começa com nenhum conhecimento prévio das regras do xadrez. Seu único objetivo é vencer um adversário de nível mestre pré-configurado. Para isso, ela deve primeiro, por tentativa e erro, **aprender as próprias regras do jogo** e, em seguida, desenvolver a estratégia necessária para atingir seu objetivo final.

A tese central é que, ao forçar a IA a deduzir as regras a partir de um feedback básico (legal/ilegal, vitória/derrota), ela desenvolverá uma compreensão mais profunda e possivelmente não-convencional do jogo.

## Conceitos Fundamentais

### 1. Aprendizado *Tabula Rasa* por Tentativa e Erro

Esta é a principal inovação. O sistema começa sem saber o que é um movimento legal.
* **Tentativa de Movimento**: A IA propõe um movimento a partir de um vasto espaço de ações possíveis.
* **Feedback do Ambiente**: O ambiente (o código) valida a ação.
    * Se o movimento for **ilegal**, a IA recebe um **sinal de penalidade imediato (-1)**, sua vez não avança, e ela é forçada a tentar novamente.
    * Se o movimento for **legal**, o jogo prossegue.
* **Recompensa Final**: O resultado da partida (vitória/derrota/empate) fornece o sinal de recompensa principal que guia o desenvolvimento da estratégia.

Este processo força a rede neural a, primeiro, atribuir probabilidades extremamente baixas a movimentos ilegais, efetivamente "aprendendo as regras", para depois se concentrar em encontrar os movimentos estratégicos.

### 2. Busca em Árvore de Monte Carlo (MCTS) em Ambiente Incerto

A MCTS continua sendo o "pensamento" profundo do sistema, mas agora opera em um ambiente de maior incerteza. A rede neural guia a busca, e a MCTS explora as sequências de lances. No entanto, a busca pode encontrar ramos que correspondem a movimentos ilegais. Esses ramos são rapidamente "podados" pela penalidade negativa, ensinando a MCTS a focar em sequências de lances válidos e promissores.

### 3. Aprendizagem Multitarefa (Multi-Task Learning - MTL)

A arquitetura de uma única rede com múltiplas "cabeças" é mantida por sua eficiência. O conhecimento adquirido para uma tarefa beneficia as outras.
* A **cabeça de valor** aprende a avaliar posições com base não apenas no resultado final, mas também no feedback imediato de movimentos ilegais.
* A **cabeça de política própria** aprende a propor movimentos que são, primeiro, legais e, segundo, estrategicamente fortes.
* A **cabeça de política do oponente** continua aprendendo a prever os lances do adversário durante o auto-jogo.

## Arquitetura do Sistema

O coração do sistema permanece uma Rede Neural Residual (ResNet) profunda com três saídas. A capacidade da rede foi aumentada para lidar com a complexidade adicional de aprender as regras do jogo.

### A Rede Neural de Três Cabeças

1.  **Cabeça de Valor (`v`)**: Avalia a posição, variando de -1 (derrota/ilegal) a +1 (vitória).
2.  **Cabeça de Política Própria (`p_self`)**: Gera uma distribuição de probabilidade sobre todo o espaço de ações. Tem a difícil tarefa de aprender a zerar a probabilidade de lances ilegais.
3.  **Cabeça de Política do Oponente (`p_opp` - O Oráculo)**: Prevê o lance mais provável do oponente.

## Processo de Treinamento: A Jornada do Mestre

O processo de treinamento foi completamente redesenhado. Não há um número fixo de iterações. O treinamento é um ciclo contínuo que termina apenas quando um objetivo é atingido.

**Objetivo Final: Vencer um motor de xadrez de nível Mestre (ex: Stockfish) em um placar geral.**

O loop de treinamento funciona da seguinte maneira:

1.  **Geração Contínua de Dados (Auto-Jogo)**: A IA joga incansavelmente contra si mesma. Cada partida gera centenas de exemplos de treinamento, incluindo dados de movimentos legais com resultados de jogo e dados de movimentos ilegais com penalidades imediatas.
2.  **Armazenamento e Treinamento**: Os dados são armazenados em um `ReplayBuffer`. A rede neural é continuamente treinada com amostras desse buffer, aprimorando sua compreensão das regras e da estratégia.
3.  **Avaliação Cíclica Contra o Mestre**: Periodicamente, o treinamento pausa, e a versão atual da Quimera joga uma série de partidas contra o motor de xadrez "Mestre" configurado.
4.  **Verificação do Objetivo**: O placar geral (`vitorias_quimera` vs. `vitorias_mestre`) é atualizado.
5.  **Repetição**: Se a Quimera ainda não tiver um placar superior, o ciclo de Geração de Dados e Treinamento recomeça. O processo só para quando `vitorias_quimera > vitorias_mestre`.

## Avaliação e Pré-requisitos

### Avaliação de Força

A força da IA não é mais medida por uma classificação Elo teórica, mas sim pelo seu **desempenho prático e direto contra um adversário de ponta**. O sucesso é definido de forma binária: a Quimera superou ou não o Mestre.

### Pré-requisitos

Para executar este projeto, os seguintes componentes são necessários:

* **Linguagem**: `Python 3.8+`
* **Framework de Deep Learning**: `PyTorch`
* **Lógica de Xadrez e Interação com Motores**: `python-chess`
* **Hardware**: GPU NVIDIA com suporte a `CUDA` é **altamente recomendada** devido à enorme carga computacional.
* **Motor de Xadrez UCI (Obrigatório)**: Você **precisa** baixar um motor de xadrez que use o Protocolo de Interface Universal de Xadrez (UCI).
    * **Recomendação**: **Stockfish** (disponível em [stockfishchess.org](https://stockfishchess.org/download/)).
    * Após o download, você deve atualizar o caminho para o executável do motor na variável `MASTER_ENGINE_PATH` dentro do script.

### Como Executar

1.  **Cumpra os Pré-requisitos**: Instale Python, PyTorch e as outras bibliotecas. Baixe o Stockfish e coloque-o em uma pasta conhecida.
2.  **Configure o Caminho**: Abra o script Python e edite a variável `MASTER_ENGINE_PATH` para apontar para o local do seu executável do Stockfish.
3.  **Execute o Script**: No seu terminal, execute o comando: `python nome_do_script.py`
4.  **Inicie o Treinamento**: O programa perguntará se você deseja iniciar o treinamento. Digite `y` e pressione Enter para começar a jornada da Quimera.
