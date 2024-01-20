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
  <img src= "https://github.com/in-sukim/NLP-Paper/assets/43094223/86794316-6db3-4898-9401-05b0084d697e" align="center" width="60%" height="60%"> 
</p>

### 3.3. Robustness and Flexibility

