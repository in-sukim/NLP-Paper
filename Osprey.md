# Osprey: Pixel Understanding with Visual Instruction Tuning
Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, Jianke Zhu

## Abstract
- 최근 Multimodal large language models(MLLMs)는 visual instruction tuning을 통해 general-purpose vision- language 능력을 갖춤.
- 하지만 현재의 MLLMs는 image-level이나 box-level의 이해에 초점, pixel level에서 세밀한 vision-language alignment을 갖추기에는 부족함.
- 이 논문에서는  pixel-wise visual understanding을 위해 Osprey라는 mask-text instruction tuning approach 제안
  - integrating fine-grained mask regions into language instructions
- Mask-based region-text dataset with 724K samples
- Vision encoder: Convolutional CLIP backbone model -> pixel 수준의 representation 추출
