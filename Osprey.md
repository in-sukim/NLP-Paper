# Osprey: Pixel Understanding with Visual Instruction Tuning
Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, Jianke Zhu

## Abstract
- 최근 Multimodal large language models(MLLMs)는 visual instruction tuning을 통해 general-purpose vision- language 능력을 갖춤.
- 하지만 현재의 MLLMs는 image-level이나 box-level의 이해에 초점, pixel level에서 세밀한 vision-language alignment을 갖추기에는 부족함.
- 이 논문에서는  pixel-wise visual understanding을 위해 Osprey라는 mask-text instruction tuning approach 제안
  - integrating fine-grained mask regions into language instructions
- Mask-based region-text dataset with 724K samples
- **Vision encoder backbone model**: Convolutional CLIP<br/> -> 고해상도 입력에서 pixel 수준의 representation 추출
- Segment Anything Model(SAM)와 통합하면 더 많은 역할 가능.

## 1. Introduction
최근 LLaVA, MiniGPT-4, Otter, InstructBLIP 등과 같은 많은 MLLMs은 Instruction-following과 시각적 추론 능력에 대해 인상적인 결과를 보여주지만,<br/>
대부분 Image-level에서 image-text pair 사용하여 region-level에서 region classification, captioning, reasoning 능력 부족.

region-level에서의 능력을 향상시키기 위해 Kosmos-2, Shikra, PVIT, GPT4RoI 및 GLaMM등의 최근 실험에서는 bounding box 경계를 설정하고 <br/>object-level에서 공간적 특징을 활용하여 visual instruction tuning을 시도. 그러나 sparse bounding box를 사용하게 되면 
