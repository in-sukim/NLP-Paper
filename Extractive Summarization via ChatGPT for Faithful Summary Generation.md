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
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/27d63474-9930-4e23-9550-a8e26d47d9f1)

- Supervised summarization 모델 훈련에서는 greedy algorithm을 사용하여 extractive ground-truth labels<br/>(**ORACLE**:Optimal Recall Aware Learning for Extractive summarization)을 생성하는 것이 일반적.
- gold summary와 비교하여 ROUGE 점수를 최대화하는 여러 문장을 선택.
  - 한 번에 한 문장씩 요약에 점진적으로 추가되며, 현재 선택된 문장 집합의 Rouge 점수가 전체 gold summary에 대해 최대화.
  - 남은 후보 문장 중 어느 것도 현재 요약에 추가했을 때 Rouge 점수가 향상되지 않을 때까지 진행
  - 이렇게 선택된 문장 부분 집합을 추출적인 ground truth로 반환
### 3.2 In-context Learning
- Large Language Model은 다양한 downstream task에서 강력한 few-shot 성능. 이를 in-context learning (ICL)이라 함.
  
- 일반적인 ICL prompt는 model $M$에게 $k$개의 document-summary pair의 예시를 주고, 문서에 대한 요약 $\hat{s}$를 예측하도록 하는 방식.
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/29b4bf76-9edb-4918-929d-dbefec397681)

- 이전 연구들은 simple input-output pair 이외에도 설명과 chain-of-thought(COT)를 prompt에 포함시키는 것이 model에 benefit.
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/37286315-6544-4907-b7ae-37d3bae1bbb1)
- $C = {(d^1,e^1,s^1)...(d^k,e^k,s^k)}$는 set of input-explanation-output
- 본 논문에서는 zero-shot setting외에 extractive summarization에서 explanations의 유무에 따라 in-context learning 영향 실험.
### 3.3 Extract-abstract Summarization
- Extractive summaries를 사용하여 Abstractive summary를 생성하는 과정
- 1. 중요한 문장을 추출하여 Extractive summaries($s^E$) 생성.
- 2. Extractive summaries를 사용하여 LLM에게 요약을 생성하도록 요청.<br/>
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/83d33da6-5839-4910-950a-d3cf05a93056)<br/>
- 본 논문에서는 extract-then-generate pipeline을 통해 LLM summary genration hallucination 문제를 완화하는 것을 목표.

## 4 Experiments and Analysis

### 4.1 Experiment Settings
**Datasets**<br/>
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/1fda19dc-512b-4218-8f68-d2db440033d1" align="center" width="40%" height="40%"> 
</p>

**Evaluation**
- ROUGE: 요약 성능 평가
- G-EVAL:GPT-based evaluation metric
- FactCC: 요약의 faithfulness 평가
- QuestEval: 요약의 faithfulness 평가
- 50개의 dev set에서 best prompt를 선택하고 각 데이터셋의 test set에서 1000개의 예제를 무작위로 추출하여 평가. <br/>
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/3d07a354-462c-4145-82a9-391959902aa6" align="center" width="50%" height="50%"> 
</p>


### 4.2 Experiments Results
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/fc173e78-4039-4748-9fa0-bd32cb0811b6)
- 상단 블록 MatchSum의 SOTA 점수 포함

- 하단 블록 BRIO의 SOTA 점수 포함
- Extractive, Abstractive 방법 모두 SOTA 모델보다 낮은 ROUGE 점수를 보이지만, G-EVAL에서는 더 높은 점수.
  - CNN/DM, PubMed에서 높은 성능, 다른 두개의 abstractive dataset에서는 낮은 성능
  - 이러한 결과는 데이터셋의 reference summaries의 편향과 ROUGE 점수의 한계 때문.
- ChatGPT와 SOTA 모델간의 ROUGE 점수의 차이가 extractive 세팅일 때 더 적다.
- in-context learning과 reasoning은 extractive summarization에 좋은 영향.
- XSum 데이터에서만 in-context learning 성능 저하 관찰. XSum 데이터셋의 짧은 ORACLE 특성 때문.
- COT 방법과 함께 패턴을 더 잘 이해할 수 있어 개선되는 결과.

### 4.3 Extract Then Generate
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/d760233f-ea31-4db9-821c-6ef65d5c2788)" align="center" width="70%" height="70%"> 
</p>

-  extract-then-generate framework의 효과를 검증.
- factual consistency 크게 향상.
- ChatGPT 기반 요약에서는 summary에 new content를 만드는 경향
- 하지만 extract-then-generate framework는 문서로부터 추출된 중요한 문장을 활용하여 이러한 경향을 효과적으로 완화.
- ORACLE을 활용한 경우와 유사한 summary faithfulness 개선을 보여줌.
- ORACLE-Abs을 사용한 경우가 ChatGPUT의 ROUGE 점수 측면에서 가장 크게 향상.
- extract-then-generate framework의 경우 extractive summaries의 성능에 크게 의존
  - summary faithfulness를 효과적으로 개선하면 더 나은 성능.
  
### 4.4 Positional Bias
- extractive summarization의 Lead bias
  - 뉴스 분야: 기사 초반부 가장 중요한 정보를 포함하는 경우 많다.
  - LLM들이 extractive summarization에서 문장의 위치와 같은 superficial features에 의존할 수 있다.
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/fb12eae0-0fe3-4acf-9ff5-0f3def9e0259" align="center" width="40%" height="40%"> 
</p>


## 5 Conclusion
