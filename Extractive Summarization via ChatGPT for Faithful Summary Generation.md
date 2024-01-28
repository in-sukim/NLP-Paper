# Extractive Summarization via ChatGPT for Faithful Summary Generation
Haopeng Zhang Xiao Liu Jiawei Zhang

## Abstract
- traditional fine-tuning 방법과 다양한 benchmark dataset을 통해 ChatGPT의 extractive summarization 성능 평가.
- ROUGE score는 기존 supervised system보다 낮은 경향, 하지만 LLM 기반 평가 메트릭에서는 더 높은 성능.
- in-context learning과 chain-of-thought reasoning의 효과를 높이기 위한 연구 진행.
  
## 1. Introduction
- Document summarization은 가장 중요한 정보를 보존하면서 text를 압축하는 것을 목표.
- 공개되는 텍스트 데이터의 양이 증가함에 따라, automatic summarization 접근 방식의 중요성 또한 중요해지고 있다.
- **Summarization**
  - **Abstractive**: flexible 하고 redundant가 적은 이점이 있지만 문법에 맞지 않거나 사실이 아닌 내용을 생성할 수 있다.
  - **Extractive**: source document에서 직접 문장을 선택하여 문법적으로 맞고 source sentence에 faithful한 결과.
- 최근 (Goyal et al., 2022)의 연구에서는 더 낮은 Rouge 점수에도 불구하고 인간 주석자들은 GPT-3가 생성한 텍스트를 선호하는 결과.
- 인간이 작성한 뉴스 요약 결과와 LLM 모델의 결과가 비교 가능하다는 결과(Yang et al., 2023; Luo et al., 2023)
- 하지만 이러한 연구들은 Abstractive Summarization 방식에 초점.
- hallucination 문제를 해결하기 위해서라도 LLM을 활용한 **Extracive summarization system**에 대한 연구 필요성 강조.
- **Contributions**
  - ChatGPT의 extractive summarization 방법에 대해 확장하고 그 성능을 평가한 최초의 시도. 
  -  ChatGP를 사용하여 extractive summarization을 위한 n-context learning,chain-of-thought reasoning approaches 효과 실험.
  -  Extraction step을 abstractive summarization으로 확장, extract-then-generate framework를 통해 faithfulness 향상.
## 2. Related Work

## 3. Methods

### 3.1 Task Formulation

### 3.2 In-context Learning

### 3.3 Extract-abstract Summarization

## 4 Experiments and Analysis


### 4.1 Experiments and Analysis

### 4.2 Experiment Settings

### 4.3 Extract Then Generate


### 4.4 Positional Bias


## 5 Conclusion
