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
- **Osprey-724K Dataset:** SAM(Segment Anything Model)을 통해 이미지에서 클래스를 나눈 후 fine-grained semantics 생성<br/>
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