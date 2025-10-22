# Detecção de Queda para o Projeto Lucas

## 1. Introdução e Contextualização

O envelhecimento da população é uma realidade demográfica global, trazendo consigo a crescente necessidade de soluções que garantam a segurança e a qualidade de vida dos idosos. Dentre os acidentes domésticos, as quedas representam um dos maiores riscos, sendo uma das principais causas de lesões graves, hospitalizações e perda de autonomia nesta faixa etária. Um fator crítico para mitigar as consequências de uma queda é o tempo de resposta: quanto mais rápido o socorro for prestado, melhores são as chances de recuperação.

## 2. O Problema

Muitos idosos, especialmente os que vivem sozinhos, podem ficar incapacitados de pedir ajuda após uma queda. Sistemas de alerta manuais (como botões de pânico) dependem da capacidade do idoso de acioná-los, o que pode não ser possível em caso de perda de consciência ou imobilização. Portanto, existe uma demanda urgente por um sistema de monitoramento automático, não invasivo e inteligente.

## 3. A Solução Proposta

Este projeto propõe o desenvolvimento de um sistema de "Assistente de Idosos" focado na detecção automática de quedas, utilizando visão computacional e processamento de imagem. O sistema analisará um fluxo de vídeo (proveniente de câmeras estrategicamente posicionadas no ambiente) em tempo real para identificar eventos de queda. Ao detectar um incidente, o sistema será capaz de disparar um alerta imediato para cuidadores, familiares ou serviços de emergência.

## 4. Metodologia e Tecnologias

O núcleo tecnológico do projeto será construído sobre duas ferramentas principais:

1.  **Python:** Como linguagem de programação principal, o Python oferece um ecossistema robusto e maduro para aprendizado de máquina e processamento de imagem. Bibliotecas como OpenCV (para manipulação de vídeo e imagem), NumPy (para operações numéricas) e outras frameworks de comunicação (para o envio de alertas) serão centrais para o desenvolvimento.

2.  **YOLO (You Only Look Once):** Utilizaremos o YOLO, um algoritmo de detecção de objetos em tempo real de última geração. O YOLO é conhecido por sua alta velocidade e precisão. Neste projeto, ele será treinado ou ajustado (fine-tuned) para:
    * **Detecção de Pessoas:** Identificar a presença de um ou mais indivíduos no quadro de vídeo.
    * **Análise de Postura/Movimento:** O sistema não irá apenas "ver" a pessoa, mas também analisar as características de sua *bounding box* (caixa delimitadora) ao longo do tempo. Uma queda é tipicamente caracterizada por uma rápida mudança na posição vertical e uma subsequente mudança na proporção da *bounding box* (de vertical para horizontal), seguida de um período de imobilidade.

O fluxo de trabalho básico consistirá em capturar frames de vídeo, passá-los pelo modelo YOLO para detecção e, em seguida, aplicar uma lógica de processamento de imagem e análise temporal para classificar se os movimentos detectados configuram uma queda.

## 5. Objetivos e Impacto Esperado

O objetivo principal é criar um sistema confiável, preciso e de baixo custo que possa operar continuamente. Ao automatizar a detecção de quedas, este projeto visa aumentar significativamente a segurança de idosos em ambiente domiciliar, proporcionar tranquilidade aos familiares e cuidadores e, o mais importante, reduzir o tempo crítico entre um incidente e o recebimento de ajuda.

## 6. Arquivos

```text
.
├── database : está vazio, precisa baixar
│   ├── file1.txt
│   └── file2.md
├── deteccao_queda
├── detectar_keypoints
├── passo1
│   └── detecta_posicao_mediapipe.py   : exemplo de detecção de corpo usando Mediapipe
│   └── detecta_posicao_yolo.py        : exemplo de detecção de corpo usando Yolo
├── LICENSE
└── README.md
```

## 7. Links

- Artigo em produção: https://pt.overleaf.com/project/68aca2b213569069572e3a09
- Projeto pai: https://github.com/VRI-UFPR/lucas