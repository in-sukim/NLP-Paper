# FRUIT: Faithfully Reflecting Updated Information in Text
Robert L. Logan IV, Alexandre Passos, Sameer Singh, Ming-Wei Chang

## Abstract
- 위키피디아와 같은 **텍스트 지식 베이스를 최신 상태로 유지**하기 위해 **외부 지식에 기반**한  **편집제안 문제**는 
충분히 탐구 되지 않았다.
- 이 논문에서는 **텍스트에서 업데이트된 정보를 충실히 반영**하여 **기존 문서를 업데이트**하는  것을 목표로 하는 **FRUIT** 방법 제안.
- **FRUIT-WIKI dataset**: Wikipedia snapshot 쌍에서 생성된 17만 개 이상의 distantly supervised data로 구성된 데이터
- 편집용 T5 기반 모델인 **EDIT5**를 소개.

## 1. Introduction

- 스포츠, 엔터테인먼트, 일반 지식 등 분야에서 **정보**는 **동적으로 변화**
- 이러한 **변화된 정보**를 **반영**하여 **텍스트 기반 지식 베이스를 유지**하는 노력 필요
    - 영어 위키백과에서 12만명의 인원이 분당 약 120회의 편집을 하고 매일 600개의 새로운 문서 작성 <br />→ 지식 베이스가 확장됨에 따라 유지보수 작업의 양도 증가하여 정보의 일관성을 유지하는데 어려움
- 다양한 선행 연구들이 있었지만 외부 지식에 기반한 편집을 제안하는 문제는 충분히 탐구되지 않았다.<br />→ **기존 텍스트**에 **새로운 정보를 통합**하는 "**FRUIT**"라는 새로운 텍스트 생성 작업 제안
  ![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/d2a886bb-2c5b-438a-93ef-b0d28d80a2e9)

- **도전과제**
  - 새로운 정보가 기존 정보와 모순되는 경우, 기존 정보를 우선시해야 한다.
  - 생성된 텍스트는 원본 문서와 새로운 정보를 모두 반영,기존 정보와 새로운 정보가 모순되는 경우를 제외하고는 일관성을 유지해야 한다. <br />→ 따라서 이 작업은 다중 문서 요약과 데이터-텍스트 생성 태스크에서 발생할 수 있는 도전 과제의 결합 유형

- **FRUIT-WIKI**
    - 자동화된 프로세스를 통해 생성된 17만개 데이터. 위키피디아 스냅샷을 비교하여 문서 업데이트를 식별하고 파이프라인을 통해 업데이트-기존 데이터 쌍 데이터 생성
    - Silver Dataset이기 때문에 human annotated된 914개의 Gold Dataset 구축
    - Silver Data로 모델을 훈련하고 검증한 후, Gold Dataset으로 최종 성능 평가
 
## 2. The FRUIT Task

**2.1 Task Definition**
원본 텍스트에 대한 새로운 정보의 컬렉션이 주어지는 경우,목표는 원본 텍스트를 새로운 정보를 반영하도록 업데이트하는 것. <br />
- 특정 주제 $A$에 대해 시점 $t,t'$에 작성된 텍스트 쌍 $A^t, A^{t'}$<br />
- t 시점과 t' 사이의 새로운 정보,즉 Evidence $\mathcal{E}^{t \rightarrow t'} = {E_1, .. E_{|\mathcal{E}|}}$ <br />
- 새로운 증거는 구조화된 객체와 그렇지 않은 텍스트가 포함될 수 있다.
- $A^t$와 $\mathcal{E}^{t \rightarrow t'}$가 주어졌을 때, 업데이트 된 $A^{t'}$를 생성하는 것이 목표
