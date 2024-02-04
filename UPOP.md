# Unifying Vision, Text, and Layout for Universal Document Processing

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang,Mohit Bansal1

# Abstract
- text, image, layout modalities를 통합하여 document understanding, generation 등 다양한 task format을 수행할 수 있는 foundation Document AI model **Universal Document Processing(UDOP)** 제안.
- textual content와 document image 사이의 spatial correlation을 통해 하나의 **uniform representation**으로 모델링
- 새로운 Vision-Text-Layout Transformer을 통해 pretraining과 multi-domain downstream tasks를 **prompt-based sequence generation** 체계로 통합


# 1. Introduction
- diverse document type과 format에 따라 구조적으로 다른 위치에 분포하기 때문에 문서의 layout을 이해하는 것 중요.

- text와 visual modalities 사이의 crossmodal interaction은 regular
vision-language data보다 강하다
    - 문서 내의 텍스트가 이미지 안에 시각적으로 표현
- 이러한 특성으로 두가지의 challenges 제기
    - image, text and layout modalities 간의 강한 correlation를 어떻게 활용하여 전체 document를 모델링할 것인가
    - 다양한 vision, text, layout tasks를 효과적으로 학습하고 다른 domain에 적용 할 것인가

- **Contributions**
    1. **Unified representations and modeling** for **vision, text and layout modalities** in document AI.
    2. Unified all document tasks to the **sequence-to-sequence generation framework**.
    3. Combined novel **self-supervised objectives** with supervised datasets in pretraining for unified document pretraining.
    4. **UDOP can process and generate text, vision, and layout modalities together**, which to the best of our knowledge is first one in the field of document AI.
# 2. Related Work
- **Unifying Model Architectures in Multimodal Learning.**
    - concatenates text token embeddings와 projected image patches를 multimodal Transformer의 입력으로 사용.
    - 각 modality를 각각 encode하는 two-tower, three-tower architecture
    - two-tower architecture 위에 projection heads, fusion network을 통해 multimodal representations 생성.
- **Unifying Tasks with the Generative Framework.**
    - finetunes language models with instructions on 1.8k tasks
    - vision-language tasks by converting training objectives to sequence generation.
    - combines more tasks(image generation), by converting images and bounding boxes to discrete tokens.

- **Document Artificial Intelligence.**
    - **LayoutLM**: masked language modeling과 document classification task를 통해 2D positional information과 image embeddings이 통합된 문서에 대해 BERT 모델 pretrain
    - **Visual-Language BERT** 유사한 아키텍처를 채택, masked image/region modeling 제안. layout 정보에서 읽는 순서를 활용
    - CNN을 통해 추출한 region feature와 sentence-level text representation을 사용하여 **multimodal encoder** 모델링, **self-supervised objectives**로 훈련.
    - OCR을 사용하지 않고 document image에서 textual output generate
    - generative training objectives를 사용하여 **un-labeled,labeled document data**에서 **generative language model** 훈련.
    - 문서를 **tokens bounding box**를 통해 모델링하는 방법 제안.

# 3. Universal Document Processing
![Alt text](image.png)
- unified learning objectives
- **Vision-Text-Layout Transformer**
    - model architecture for text, vision, and layout
- document image $v$가 주어졌을 때, OCR을 통해  문서 내 text token {$s^i$}인식, 각 text token에 대한 bounding box {($x^1_i, y^1_i, x^2_i, y^2_i$)}
- Input: $(v, \{s_i\}_{i=1}^M, \{(x_i^1, y_i^1, x_i^2, y_i^2)\}_{i=1}^M)$
![Alt text](image-1.png)

## 3.1. A Unified Vision, Text, and Layout Encoder


## 3.2. Vision-Text-Layout Decoder

# 4. Unified Generative Pretraining


## 4.1. Self-Supervised Pretraining Tasks

## 4.2. Supervised Pretraining Tasks


# 5. Experimental Setup

## 5.1. Model Pretraining


## 5.2. Downstream Evaluations


# 6. Analysis


## 6.1. Visualization Analysis


## 6.2. Ablation Analysis


## 6.3. Effectiveness of the Vision Modality


# 6.3. Effectiveness of the Vision Modality