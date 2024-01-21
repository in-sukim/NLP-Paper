# Osprey: Pixel Understanding with Visual Instruction Tuning
Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, Jianke Zhu

## Abstract
- 최근 Multimodal large language models(MLLMs)는 visual instruction tuning을 통해 general-purpose vision- language 능력을 갖춤.
- 하지만 현재의 MLLMs는 image-level이나 box-level의 이해에 초점, pixel level에서 세밀한 vision-language alignment을 갖추기에는 부족함.
- 이 논문에서는  pixel-wise visual understanding을 위해 Osprey라는 mask-text instruction tuning approach 제안
  - integrating fine-grained mask regions into language instructions
- Mask-based region-text dataset with 724K samples
- **Vision encoder backbone model: Convolutional CLIP**<br/> -> 고해상도 입력에서 pixel 수준의 representation 추출
- Segment Anything Model(SAM)와 통합하면 더 많은 역할 가능.

## 1. Introduction
- 최근 LLaVA, MiniGPT-4, Otter, InstructBLIP 등과 같은 많은 MLLMs은 Instruction-following과 시각적 추론 능력에 대해 인상적인 결과를 보여주지만,<br/>
대부분 Image-level에서 image-text pair를 사용하여 region-level에서 classification, captioning, reasoning 능력 부족.

- region-level에서의 능력을 향상시키기 위한 최근 실험에서는 bounding box 경계를 설정하고 object-level에서 공간적 특징을 활용하여 <br/>visual instruction tuning을 시도.
- 그러나 sparse bounding box를 사용하게 되면 관계없는 배경 feature를 포함하게 되고, LLM의 visual instruction을 위한 <br/>region-text pair alignment의 부정확성을 야기할 수 있다.<br/>

<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/5c3de6cc-3f3a-4515-a2be-4aac96a0a921" align="center" width="50%" height="50%"> 
</p>

- 세밀한 픽셀 수준의 이해를 위해 MLLMs의 능력을 확장할 수 있는 "**Osprey**" 제안
- 세밀한 visual mask extractor를 통해 추출한 시각적 특성을 language instructions과 함께 교차 결합하여 LLM의 input sequence로 사용.
- 고해상도 Input의 사용을 용이하게 하기 위해 Vision Encoder backbone model로 CLIP을 사용.
- Convoutional CLIP은 효율성과 강건함을 유지하면서 더 큰 입력 해상도에 대한 일반화 성능 우수.
- **Osprey-724K Dataset:** large- scale mask-based region-text dataset. 상세한 설명과 대화뿐만 아니라 풍부한 속성 정보도 포함<br/>
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/d4d77b61-de7c-4e8b-b8c1-09ec5aeebf75" align="center" width="50%" height="50%"> 
</p>

### Contributions
- Multimodal large language models을 pixel-level instruction tuning이 가능하게 하는 "Osprey" 제안
- mask-text pairs로 이루어진 Large-scale instruction tuning dataset "Osprey-724K" 구축
- fine-grained visual understanding approach로서 다양한 region understanding task에서 SOTA 달성.

## 2. Related Work
- 최근 연구는 visual instruction tuning을 위해 pre-trained LLM을 어떻게 활용할 수 있을지에 집중
- 대표적인 모델들은 visual input encoding을 위한 pre- trained visual backbone과 사용자 지시를 이해하고 응답을 생성하기 위한 LLM,<br/> 그리고 vision-language cross-modal connector로 이루어진 아키텍처.
- Image-level에서는 좋은 성능을 보였지만, 특정 영역을 참조해야 하는 경우 제한적인 성능.
- Segment Anything Model (SAM)은 zero-shot 상태에서 뛰어난 segmentation 성능을 보였지만, vanilla SAM 모델은 semantic label 제공 못함.
- SEEM, HIPIE, Semantic SAM 등 다양한 접근 방식을 통해 확장된 모델들은 category를 예측할 수 있지만 색상, 위치, 설명을 하지 못해 실제 응용해서 사용하기에는 부족.
- GPT4RoI, PVIT, Kosmos-2, Shikra, Ferret,GLaMM 등의 MLLMs에서는 region-based image 이해가 가능하게 했지만, <br/>bounding bax를 참조 영역으로 사용하여 정확하지 않은 region-text alignment를 야기할 수 있다.

## 3. Osprey-724K Dataset
- 사전에 구성된 프롬프트 템플릿을 사용하여 GPT-4를 활용해 mask-text pair 생성.
- response의 robustness와 flexibility를 위해 짧은 형식의 response formatting prompt와 함께 negative sample mining method 사용.<br/>
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/7b417bff-2b23-4905-a21d-f32e97372af0" align="center" width="80%" height="70%"> 
</p>
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/8bb56fcd-5326-42b4-bebd-0210371b6295" align="center" width="80%" height="70%"> 
</p>

### 3.1. Object-level Instructions
- COCO, Ref-COCO, RefCOCO+, RefCOCOg 데이터의 경우 image-level에서 object-level의 caption 제공
- 하지만 이러한 caption들은 짧고 의미 있는 맥락 정보를 포함하지 않아 MLLMs를 훈련하는데 적합하지 않음.
- 이 문제를 해결하기 위해 object category, object type, object action, location, color, status 등의 정보를 담은 <br/>fine-grained region-based instruction data를 생성하는 파이프라인 구축
- COCO 데이터에서 bounding box, region caption 정보를 사용하고, GPT-4와 같은 언어모델을 통해 LLaVA-115K에서 COCO 데이터와 유사한 상황을 자세하게 설명하는 description과 결합하여 새로운 description과 대화 샘플 생성.
- 총 197K개의 unique object- level mask-region instruction-following samples 수집

### 3.2. Part-level Instructions
- PACO-LVIS 데이터셋은 75개 object의 456가지 특정 부분 클래스를 포함.
- 색상, 패턴, 재료, 반사도 수준 등의 다양한 속성 포함. 이를 통해 object에 대한 자세한 이해 가능
- PACO-LVIS 데이터셋에서 추출된 정보를 토대로 Context를 입력으로 주고 사전에 정의된 질문들을 통해 QA 형식의 306K개의 데이터셋 구성
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/2e953af9-00aa-441c-b591-98fa8805c663" align="center" width="60%" height="60%"> 
</p>

### 3.3. Robustness and Flexibility
#### Robustness
- 이전 연구에서 MLLMs는 object hallucination문제 발견.
- 이를 완화하고자 Positive/Negative Sample을 구성.
- 특정 영역이 특정 카테고리에 속하는지 여부를 질문하는 쿼리 -> "예/아니오"
- "아니오"에 해당하는 경우를 Negative Sample에 해당.
- 공간적으로 가장 가까운 obejct의 카테고리를 식별하게 하여 어떤 객체가 다른 객체와 가까이 위치해 있는지 파악하고, 객체 간의 공간 관계를 이해하는 도움.
- negative 카테고리는 target class name과 높은 semantic similarities를 가진 카테고리로 선택.
- SentenceBert 기법을  통해 semantic similarities 평가
- 모델이 특정 카테고리를 잘못 인식하는 문제를 완화하는 데 도움. 후보 상위 8 개 중 하나를 임의로 선택.
- LVIS 데이터셋에도 이러한 방법을 적용하여 동작.
<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/52f59d62-85e6-4678-9cad-813ad54a559c" align="center" width="50%" height="50%"> 
</p>

#### Flexibility
- 짧은 형태의 response instruction 구성.
- 간단한 답변을 요구할 때 질문 끝네 명시적으로 짧은 형태의 프롬프트 추가 
## 4. Method of Osprey
### 4.1. Model Architecture
- Osprey는 image-level vision encoder, pixel-level mask-aware visual extractor, LLM으로 구성.

<p align="center">
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/cdb886d5-33e0-4fab-bb99-2a6a3c051605" align="center" width="70%" height="70%"> 
</p>

### 4.1.1 Convolutional CLIP Vision Encoder
- ViT 기반 CLIP 모델은 224x224 또는 336x336의 해상도 채택. 그러나 이러한 해상도에서 픽셀 수준의 representations를 사용하여 <br/>fine-grained image understanding 어려움.
- 이를 해결하기 위해 convolutional CLIP model(ResNet, ConvNeXt) 도입.
- 다양한 input 해상도에서 높은 일반화 성능.

### 4.1.2 Mask-Aware Visual Extractor
- 각 object region의 pixel-level feature를 추출하기 위해 mask-aware visual extractor 제안.
- mask-level에서 visual feature를 추출하는데 그치지 않고, 각 영역 $R_i$의 공간적인 위치 정보도 추출.
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/18fb123e-f11d-4205-b35b-3ee9aa628dd9)
- Vision Encoder 출력 feature map $Z(x)_i$에 대한 mask-pooling operation
- $V_{ij}$는 객체 영역 $R_i$에 대한 해당 레벨 $j$의 시각적 특징. <br/>다중 레벨 특징을 활용하여 객체 영역의 다양한 시각적 특징을 종합적으로 분석하고 해당 객체에 대한 이해를 높인다.
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/1d1ec6ff-ccab-4b7b-b193-15557ec4d8f5)
- $V_{ij}$를 linear projection layer $P_j$를 통해 전달하여 region-level embeddings 생성.
- 다양한 이미지 레벨에서 얻은 representation Add(4개 레벨)
- MLP layer 통과하여 visual mask token $t_i$ 생성
- 각 픽셀이 해당 객체 영역에 속하면 1 아니면 0을 나타내는 정보를 담은 binary mask $M$
- 224x224 크기로 조정. Flatten하여 1차원 벡터로 변형. linear projection을 통해 spatial token $s_i$생성.
- visual mask token $t_i$와 spatial token $s_i$ 결합하여 각 mask region에 대한 최종 임베딩 생성.
- object region의 시각적 정보와 pixel-level spatial 정보를 통합하여 객체를 정확하게 인식하고 이해하는데 도움.
