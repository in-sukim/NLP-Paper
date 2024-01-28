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
  -  ChatGP를 사용하여 extractive summarization을 위한 in-context learning, chain-of-thought reasoning 효과 실험.
  -  Extraction step을 abstractive summarization으로 확장, extract-then-generate framework를 통해 faithfulness 향상.
## 2. Related Work
- 대부분의 연구에서는 Extractive summarization을 sequence classification 문제로 정의하고 sequential neural model과 다양한 encoder 적용.

- Encoder로는 recurrent neural networks와 pre-trained language models
- 다른 연구에서는 node classification 문제로 정의하고 graph neural networks를 적용하여 inter-sentence dependencies를 모델링.
- 일부 연구에서는(Brown et al., 2020) LLM을 사용. (Goyal et al.2022)에서는 LLM의 경우가 ROUGE 점수는 낮지만 human evaluators는 오히려 선호하는 결과.
- 본 연구는 **ChatGPT의 추출적 요약**에 대한 **적용 가능성을 조사**하고, **추출적인 방법이 추상적 요약의 충실도를 향상시킬 수 있는지 검토**하는 것을 목표
## 3. Methods

### 3.1 Task Formulation
- $n$개의 문장으로 구성된 문서 $d$가 주어졌을 때, Extractive summarization의 목표는 모델 $M$을 통해 요약 s를 구성하는 <br/>$m(m	\ll n)$개의 문장을 직접 추출하여 생성하는 것.
- 대부분의 기존 연구에서는 sequence labeling problem으로 정의하며, 모델 $M$은 문장이 요약 s에 포함되어야 하는 확률에 기반하여 문장 선택.
- $\hat{s}$ = arg max $pM$ (s|d).
- Supervised summarization 모델 훈련에서는 greedy algorithm을 사용하여 extractive ground-truth labels<br/>(**ORACLE**:Optimal Recall Aware Learning for Extractive summarization)을 생성하는 것이 일반적.
- gold summary와 비교하여 ROUGE 점수를 최대화하는 여러 문장을 선택.
  - 한 번에 한 문장씩 요약에 점진적으로 추가되며, 현재 선택된 문장 집합의 Rouge 점수가 전체 gold summary에 대해 최대화.
  - 남은 후보 문장 중 어느 것도 현재 요약에 추가했을 때 Rouge 점수가 향상되지 않을 때까지 진행
  - 이렇게 선택된 문장 부분 집합을 추출적인 ground truth로 반환
### 3.2 In-context Learning
- Large Language Model은 다양한 downstream task에서 강력한 few-shot 성능. 이를 in-context learning (ICL)이라 함.
- 일반적인 ICL prompt는 model $M$에게 $k$개의 document-summary pair의 예시를 주고, 문서에 대한 요약 $\hat{s}$를 예측하도록 하는 방식.
- $\hat{s}$ = argmax $pM(s|d, {d^1,s^1)...(d^k,s^k)})$
- 이전 연구들은 simple input-output pair 이외에도 설명과 chain-of-thought(COT)를 prompt에 포함시키는 것이 model에 benefit.
- $\hat{s}$ = argmax $pM(s|d,C)$
- $C = {(d^1,e^1,s^1)...(d^k,e^k,s^k)}$는 set of input-explanation-output
- 본 논문에서는 zero-shot setting외에 extractive summarization에서 explanations의 유무에 따라 in-context learning 영향 실험.
### 3.3 Extract-abstract Summarization

## 4 Experiments and Analysis


### 4.1 Experiments and Analysis

### 4.2 Experiment Settings

### 4.3 Extract Then Generate


### 4.4 Positional Bias


## 5 Conclusion
