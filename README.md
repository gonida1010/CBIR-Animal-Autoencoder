# 🐾 Animal Image Retrieval (CBIR) using Convolutional Autoencoder

본 프로젝트는 Convolutional Autoencoder(CAE)를 활용하여 동물 이미지의 잠재 특징(Latent Features)을 추출하고, 이를 기반으로 유사한 이미지를 검색하는 **내용 기반 이미지 검색(Content-Based Image Retrieval, CBIR)** 시스템을 구현한 과제입니다.

---

## 1. 프로젝트 개요
* **목표**: 이미지의 단순 픽셀 비교를 넘어, 모델이 학습한 고차원적인 특징(색상, 형태, 패턴)을 기반으로 시각적으로 유사한 동물을 검색합니다.
* **데이터셋**: [Kaggle CBIR Dataset](https://www.kaggle.com/datasets/theaayushbajaj/cbir-dataset) (호랑이, 표범 등 다양한 동물 이미지 포함)
* **핵심 기술**: Python, PyTorch, Convolutional Autoencoder, K-Nearest Neighbors (KNN)

---

## 2. 시스템 아키텍처
본 시스템은 크게 특징 추출기 학습(Training)과 유사 이미지 검색(Retrieval)의 두 단계로 구성됩니다.

### 🏗 모델 구조 (Convolutional Autoencoder)


* **Encoder**: 입력 이미지($128 \times 128 \times 3$)를 3개의 Convolutional Layer를 거쳐 256차원의 잠재 벡터로 압축합니다. 이 벡터는 이미지의 핵심 정보를 담고 있는 '지문' 역할을 합니다.
* **Decoder**: 압축된 벡터를 다시 원래 크기의 이미지로 복원(Reconstruction)하며 학습을 진행합니다. 이 과정을 통해 인코더는 이미지의 중요한 특징만을 보존하는 법을 배웁니다.
* **Bottleneck**: 256차원의 선형 레이어를 배치하여 특징 데이터를 극도로 압축, 검색 효율성을 높였습니다.

---

## 3. 주요 구현 세부 사항
* **KaggleHub 데이터 로드**: `kagglehub` API를 사용하여 데이터셋을 로컬 환경에 직접 다운로드함으로써 구글 드라이브 동기화 오류 및 압축 손상 문제를 해결했습니다.
* **데이터 전처리**: 
    * $128 \times 128$ 이미지 리사이징
    * 채널별 정규화(Normalization) 적용
* **유사도 측정 (Cosine Similarity)**: 특징 벡터 간의 거리를 계산할 때 유클리드 거리보다 이미지 특징 비교에 더 효과적인 **코사인 유사도**를 적용하여 검색 정확도를 높였습니다.
* **시각화**: Query(입력) 이미지와 검색된 상위 11개의 결과(Rank 1~11)를 거리(Distance) 정보와 함께 그리드 형태로 출력하여 성능을 직관적으로 확인할 수 있게 구현했습니다.

---

## 4. 학습 결과 및 성능
* **Loss Function**: MSE (Mean Squared Error)
* **Optimizer**: Adam ($Learning Rate = 0.001$)
* **Epochs**: 15 Epochs 수행 결과 Loss가 안정적으로 수렴하였으며, 결과 시각화 시 동일 종 혹은 유사한 색감의 이미지가 정확하게 검색되는 것을 확인했습니다.

---

## 5. 실행 방법 (Usage)

### 환경 설치
```bash
pip install kagglehub torch torchvision matplotlib tqdm torchsummary
```

### 코드 실행 순서
- Data Loading: kagglehub를 통해 /root/.cache/kagglehub/... 경로에 데이터셋 다운로드

- Training: 오토인코더 모델 학습 실행 (15 Epochs)

- Feature Extraction: 학습된 인코더로 모든 이미지의 특징 벡터 추출

- Retrieval: search_and_show(index) 함수를 사용하여 검색 결과 시각화

## 6. 결론 및 고찰

오토인코더 기반의 CBIR은 별도의 라벨링 없이도(Unsupervised) 이미지의 시각적 유사성을 파악할 수 있다는 장점이 있습니다. 추후 ResNet과 같은 사전 학습된(Pre-trained) 모델을 백본으로 사용하거나, 더 깊은 Bottleneck 구조를 설계한다면 훨씬 더 정교한 검색 시스템으로 발전시킬 수 있을 것입니다.
