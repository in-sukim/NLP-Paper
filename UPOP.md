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
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/ffc892bd-ffd1-4845-90f4-2ff6a25ba401)
- unified learning objectives
- **Vision-Text-Layout Transformer**
    - model architecture for text, vision, and layout
- document image $v$가 주어졌을 때, OCR을 통해  문서 내 text token {$s^i$}인식, 각 text token에 대한 bounding box {($x^1_i, y^1_i, x^2_i, y^2_i$)}
- Input: ![equation](https://github.com/in-sukim/NLP-Paper/assets/43094223/17c50855-57c7-4fd6-a0ec-e21e988069c1)<br/>
![image](https://github.com/in-sukim/NLP-Paper/assets/43094223/4afb478c-5e80-4195-9a2f-5c6b763c61a2)


## 3.1. A Unified Vision, Text, and Layout Encoder
-  document images, text is embedded inside the image
    - text and image pixels have one-to-one correspondence
    - 이 관계를 이용하기 위해 layout 정보를 기반으로 image pixel과 text token을 동적으로 융합하는 **Vision-Text-Layout(VTL) 아키텍처 제안**
- Document image $v \in \mathbb{R}^{H \times W \times C}$
- 이미지 내 $M$개의 word token $\{s_i\}_{i=1}^M$
- Extracted layout structure $\{(x_i^1, y_i^1, x_i^2, y_i^2)\}_{i=1}^M$
- Document image $v$를 $P \times P \times C$ 크기인 $\frac{H}{P} \times \frac{W}{P}$ image patch로 분할 
- 각 patch를 $D$-dim vector로 encode, 모든 patch embedding을 vector sequence $\{v_i\ \in \mathbb{R}^D \}_{i = 1}^N$ 그룹화
    - $N = \frac{H}{P} \times \frac{W}{P}$
- vocabulary 검색을 통해 $D$-dim numerical embedding $\{s_i\}_{i=1}^M$으로 변환.
- **Layout-Induced Vision-Text Embedding.**<br/>
    ![Alt text](image-2.png)
    - text token embedding $s_i$
    - indicator function $\phi$
    - joint representation $s'_i = s_i + v_j$, where $\phi(s_i, v_j) = 1.$
    - image patch $v_j$에 text token이 없으면 $∀i, \phi(s_i, v_j) = 0$, 
        - joint representation $v'_j = v_j$.
    - 각 image patch들은 이미 text embedding과 통합되어 있기 때문에 text token을 포함한 image patch에 지정된 joint representation이 없다.
    -  layout과 text representation을 통합시키는 방법으로  최근 연구 방법인 generative object detection 방법을 통해  layout modality를 discretize
        - [0,1] 사이 값으로 normalize된 bounding box $(x^1_i, y^1_i, x^2_i, y^2_i)$ 가정
        - bounding box (0.1, 0.2, 0.5, 0.6)
        - vocabulary size: 500
        - layout token: <50><100><250><300>
    - layout token은 text context에 삽입되고, layout generation task에 사용된다.
    - **Position Bias.**
        - 문서 내 text token의 2D text token position을 encode하기 위해 TILT(Transforming Image into Layout Token) 방식 채택
        - T5 모델에서 사용된 relative attention bias와 유사한 방식으로 text token들 사이의 상대적 위치를 모델이 이해하는데 도움.
        - 다른점은 1D position embedding을 사용하지 않는 점.
            - joint embedding과 2D position bias가 이미 layout 구조를 포함하기 때문

## 3.2. Vision-Text-Layout Decoder
- **VTL decoder**는 vision, text, and layout modalities를 함께
생성하도록 설계. text-layout decoder와 vision decoder로 구성.
- **Text-layout decoder**
    - generate text, layout token을 sequence-to-sequence로
    생성하기 위한 uni-directional Transformer decoder
- **Vision decoder**
    - masked autoencoder의 decoder를 채택하여 text와 layout information을 포함한 image pixel을 생성.
- 두 decoder 모두 VTL encoder에 cross-attend 수행.
# 4. Unified Generative Pretraining
- task prompt을 포함한 universal generative task format을 만들어 다양한 training objectives와 dataset을 통합
- pretrain UDOP on large-scale documents with and with-out human labels
![Alt text](image-3.png)
## 4.1. Self-Supervised Pretraining Tasks
-  unlabeled documents에 대한 self-supervised learning objectives 제안
-  unlabeled document는 token-level bounding box와 document image를 포함한 OCR input
- Example: “Ship Date to Retail: Week of March 14, 1994”
- "Ship Date", "of" 두 단어 Maksing
### (1) Joint Text-Layout Reconstruction 
![Alt text](image-4.png)
- layout 내 위치와 text를 예측해야 한다.

### (2) Layout Modeling
![Alt text](image-7.png)
- Document image와 context text가 주어졌을 때 text token의 위치 예측
- masking ratio 75%로 큰 비율을 사용해 태스크가 너무 쉬워지지 않도록 설정.

### (3) Visual Text Recognition
![Alt text](image-6.png)
- image 내에서 text token의 위치가 주어졌을 때 text 예측
- Masking ratio 50%
- 제공된 bounding box 정보를 통해 image내에 text의 위치를 학습하고 bounding box를 (0,0,0,0)으로 설정하여 텍스트를 예측하는데 집중하도록 유도.
- 이를 통해 이미지 내의 텍스트와 그 위치 사이의 상관관계를 이해하고, 시각적 정보와 텍스트 정보를 함께 포함하는 임베딩을 학습하도록 도움.

### (4) Masked Image Reconstruction with Text and Layout
![Alt text](image-8.png)
- Document image내에서 text와 layout을 사용하여 reconstruct image가 목표.
- vision self- supervised learning을 위해 masked autoencoders(MAE) objective 채택
- **masked autoencoders(MAE)**
    - image patch에서 일정 비율을 Masking하고 처리되지 않은 patch를 vision decoder에 입력.
    - mean squared error 사용하고 masked patch에만 loss 적용.
#### (4.a) Cross-Attention with Character Embeddings.
- textual content는 alphabetic characters, numbers,punctuation으로 구성
- 이러한 text token에 대한 **character-level composition**은 이미지 생성에 도움.
- cross-attention을 추가하여 text token의 encoder feature와 token내 characters에 대한 embedding을 고려.
- **Text Characters Embeddings**는 Encoder를 통해 encode 되지 않는 trainable parameter.
- Cross Attention은 linear computation complexity를 증가시키지만, **image generation quality**를 크게 높인다.
#### (4.b) Image Decoding.
- joint vision_text embedding은 non-masked image patch만 포함하기 때문에 Vision Decoder에 직접적으로 feeding할 수 없다.
- 그렇기 떄문에 sequence of trainable **placeholder embedding**을 Vision Docoder에 feeding
- placeholder의 sequence 길이는 image patch의 개수와 동일.
- input document image에서 image patch가 maksing 되어 있는지를 나타내기 위한 두가지 유형의 placeholder embedding 사용.
## 4.2. Supervised Pretraining Tasks
- Self-supervised tasks는 large-scale unlabeled data를 사용하여 robust representaion을 학습
- supervised tasks는 fine-grained model supervision을 위해 labeled data를 사용.
### **Classification.**
- document type을 예측하는 task
- prompt 형식: **“Document Classification on (Dataset Name)” + text token**
- target: **document class**
- 본 연구에서는 RVL_CDIP 데이터셋 사용. 16개의 document categories.

### **Layout Analysis.**
- document 내 title, paragraph 등 entity들의 위치를 예측하는 task
- prompt: **“Layout Analysis on (Dataset Name)” + entity name**
- target: **entity를 포함하는 bounding box를 예측**하는 것.
- 본 연구에서는 PubLayNet 데이터셋 사용.

### **Information Extraction.**
- entity type과 text query의 location을 예측하는 task
- prompt: **“Information Extraction on (Dataset Name) (Text Query)”**. 
- target: **query의 각 token의 entity label, bounding box**
- 본 연구에서는 DocBank, Kleister Charity(KLC), PWD, DeepForm 데이터셋 사용

### **Question Answering.**
- Document image에 관한 질문에 답하는 task
- prompt: **“Question Answering on (Dataset Name)”, then fol- lowed by the question and all document tokens**
- target: **answer**
- 본 연구에서는 WebSRC, VisualMRC, DocVQA, InfographicsVQA, WTQ (Wik- iTableQuestions) 데이터셋 사용.

### **Document NLI.**
- Document 내 두 문장 사이의 entailment relationship 예측하는 task
- prompt **“Document Natural Language Inference on (Dataset Name)”, then followed by the sentence pair.**
- target: **Entailment” or ”Not Entailment”.**
- 본 연구에서는 TabFact 데이터셋 사용.

# 5. Experimental Setup
## 5.1. Model Pretraining
- **Model Configuration.**
    - **unified encoder**와 **text-layout decoder**는 **T5-large encoder-decoder 아키텍처**
    - **vision decoder**는 **MAE-large decoder 아키텍처**
    - **794M** trainable parameters
    - **T5 tokenizer and** embedding from **Hugging Face Transformers**
    - **extend the vocabulary** to accommodate **special tokens**(e.g., new sentinel and layout tokens)

- **Data.**
    - self-supervised learnin을 위해 larget-scale document collections IIT-CDIP Test Collection 1.0 사용.
    - OCR에 의해 추출된 text, token-level bounding box를 포함한 1100만개의 스캔된 문서

- **Curriculum Learning**
    - 낮은 해상도에서는 텍스트가 불분명할 수 있어 식별이 어려울 수 있다.
    - 따라서 낮은 해상도에서 시작해서 점진적으로 해상도를 높이면서 학습하는 방식.($224 \rightarrow 512 \rightarrow 1024$)

- **Training.**
    - Adam optimizer
    - learning rate: 5e-5
    - 1000 warmup steps
    - batch size 512
    - weight decay: 1e-2
    - $\beta_1 = 0.9$
    - $\beta_2 = 0.98$
    - 각 curriculum learning마다 1epoch로 학습.
## 5.2. Downstream Evaluations
- **FUNSD, CORD, RVL-CDIP, DocVQA** 데이터셋에 대해 결과 평가(Table 3.)
- **DUE-Benchmark**의 7개 데이터셋에 대한 결과 평가(Table 2.)
- Finetuing training details(Appendix D.6)
- Performance variance(Table 9, 10.)
- 모든 **downstream task**에서 **원본 OCR annotations 사용**.
- **FUNSD(Form Understanding in Noisy Scanned Documents)**
    - train: 149, test 50 samples
    - evaluate on the entity recognition task: **predicting the entity**
        - "question", "answer", "header", "other" for text token
        - Encoder input: "The Title"
        - Generation target: "The Title[I-Header]"
    - metric: f1-score
- **CORD (Consolidated Receipt Dataset for Post-OCR Parsing)**
    - 1000개의 영수증 샘플 포함
    - "total" 또는 "subtotal"과 같은 4개의 category에서 30개의 label을 가진 information extraction dataset
    - train: 800, validation: 100, test: 100
    - metric: f1 score
    - FUNSD와 task format 같다.
- **RVL-CDIP**
    - Document classification dataset
    - train: 320K, validation: 40K, test: 40K images
    - metric: accuracy

- **DUE-Benchmark**
    - 7개의 dataset과 3개의 domain
    - Document question answering(DocVQA, InfographicsVQA)
    - Key information extraction(KLC, PWD, DeepForm)
    - Table QA/NLI(WTQ, TabFact)
    - prompt format: Section 4.2
- **Results.**
    ![Alt text](image-9.png)
    ![Alt text](image-10.png)
    - Pretrained model들을 평가 데이터를 통해 finetune
    - UDOP model 7개의 dataset에서 SOTA 달성.
    - UPDP 모델은 open-vocabulary generative model로 모든 task에 one single model을 사용함에도 불구하고  task 별 네트워크를 활용한 분류 기반 각 baseline model보다 더 좋은 성능.
    ![Alt text](image-11.png)
    - 이미지 해상도에 따른 Curriculum Learning 결과는 해상도가 높아질수록 성능이 향상되는 모습.
    - 가장 낮은 해상도 224의 경우에도 이전 모델보다 높은 성능
    ![Alt text](image-12.png)
    - UDOP 모델은 self-supervised objectives를 통해 훈련(224 해상도)
    - Joint Text-Layout, TVL Transformer 및 제안된 self-supervised objectives의 효과 입증.
# 6. Analysis
## 6.1. Visualization Analysis
- **Masked Image Reconstruction.**
    ![Alt text](image-13.png)
    - 75%의 높은 masking 비율에도 불구하고 text와 laytout signal을 통해 문서 이미지를 고품질로 재구성.
- **Document Generation & Editing.**
    ![Alt text](image-14.png)
    - 고품질 문서 생성과 편집 가능
    - 생성된 content는 고해상도이며 글꼴, 크기, 스타일 및 방향을 원본과 일관성 있게 생성.
- **Layout Customization**
    ![Alt text](image-15.png)
    - 문서 레이아웃 편집을 수행 가능
    - 문서의 layout을 처음부터 재생성하여 편집
    - 일부 image patch를 유지하고, content의 bounding box를 변경한 후, 새로운 layout으로 문서 이미지를 재생성.

## 6.2. Ablation Analysis
![Alt text](image-16.png)
![Alt text](image-17.png)
- text-based pretraining tasks의 성능 입증
## 6.3. Effectiveness of the Vision Modality
![Alt text](image-18.png)
- Vision Modalitiy를 추가했을 경우가 더 높은 성능
# 7. Conclusion
- Document 내 layout-induced vision-text representations의 강한  **spatial correlations**을 활용하여 **Vision-Text-Layout transformer**을 통해  **vision, text, layout modality를 통합**
- 8개의 task에서 SOTA 달성히며 현재 Document Understanding Benchmark Leaderboard에서 1위.
- Document AI 처음으로 **사용자 정의가 가능한 문서 생성과 편집 가능**.
