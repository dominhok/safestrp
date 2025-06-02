# SafeStrp: Multi-Task Safety-aware Driving Perception

**A comprehensive multi-task deep learning framework for autonomous driving safety perception with MTPSL cross-task consistency.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 5/5 ✅](https://img.shields.io/badge/Tests-5%2F5%20%E2%9C%85-brightgreen.svg)](#testing)

## 🎯 **프로젝트 개요**

SafeStrp는 자율주행 안전을 위한 멀티태스크 딥러닝 프레임워크입니다. 세 가지 핵심 태스크를 동시에 처리하면서 MTPSL(Multi-Task Partial Supervised Learning) 기법을 통해 태스크 간 일관성을 학습합니다.

### 🔍 **핵심 기능**

- **🚗 Object Detection**: 29개 클래스의 주행 환경 객체 탐지
- **🛣️ Surface Segmentation**: 7개 클래스의 도로 표면 분할
- **📏 Depth Estimation**: Pixel-wise 거리 추정
- **🔗 Cross-task Consistency**: MTPSL 기반 태스크 간 일관성 학습

## 🏗️ **아키텍처**

```
SafeStrp Architecture
├── 🧠 Backbone: ResNet-50 기반 DSPNetBackbone
│   ├── C3 Features (512 channels, 1/8 scale)
│   ├── C4 Features (1024 channels, 1/16 scale)
│   └── C5 Features (2048 channels, 1/32 scale)
├── 📦 Detection Head: SSD-style with 22,516 anchors
├── 🎨 Surface Segmentation Head: FCN-style upsampling
├── 📐 Depth Regression Head: Skip connections for detail
└── 🔗 Cross-task Projection Head: 512-dim embeddings
```

### 📊 **모델 사양**

| 구성요소 | 세부사항 |
|---------|----------|
| **파라미터 수** | ~60.3M |
| **FPS 성능** | 44+ images/sec (GPU) |
| **입력 크기** | 512×512×3 |
| **앵커 수** | 22,516 (7 levels) |
| **Cross-task 임베딩** | 512차원 |

## 🚀 **설치 및 환경 설정**

### 📋 **요구사항**

```bash
Python >= 3.8
PyTorch >= 1.8.0
CUDA >= 11.0 (GPU 사용 시)
```

### 🔧 **설치**

1. **레포지토리 클론**
```bash
git clone https://github.com/username/safestrp.git
cd safestrp
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

## 🎮 **사용법**

### 📚 **기본 사용**

```python
from src.core.model import ThreeTaskDSPNet

# 모델 생성
model = ThreeTaskDSPNet(
    num_detection_classes=29,
    num_surface_classes=7,
    input_size=(512, 512),
    enable_cross_task_consistency=True
)

# 추론
import torch
input_image = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    outputs = model(input_image)

print("출력 키들:", list(outputs.keys()))
# ['detection_cls', 'detection_reg', 'surface_segmentation', 
#  'depth_estimation', 'cross_task_embeddings']
```

### 🏋️ **학습**

```python
from src.losses.multitask import UberNetMTPSLLoss

# 손실 함수 설정
loss_fn = UberNetMTPSLLoss(
    detection_weight=1.0,
    surface_weight=0.5,
    depth_weight=1.0,
    cross_task_weight=0.1
)

# 손실 계산
losses = loss_fn(predictions, targets, task_mask)
total_loss = losses['total']
```

## 🧪 **테스트**

### ✅ **포괄적 테스트 시스템**

SafeStrp는 5단계의 체계적인 테스트 시스템을 제공합니다:

| 테스트 | 성공률 | 검증 내용 |
|--------|--------|----------|
| **1️⃣ Import & 의존성** | **100%** | 모든 모듈 import 및 순환 의존성 |
| **2️⃣ 모델 구조** | **100%** | 모델 생성, 파라미터, Forward pass |
| **3️⃣ 손실 함수** | **100%** | 모든 손실 함수 정확성 및 안정성 |
| **4️⃣ 추론 파이프라인** | **100%** | 성능, 메모리, 안정성, 후처리 |
| **5️⃣ Cross-task** | **100%** | MTPSL 일관성, 임베딩, 이론적 검증 |

### 🏃 **테스트 실행**

```bash
# 전체 테스트 실행
python tests/run_all_tests.py

# 개별 테스트 실행
python tests/test_1_imports_dependencies.py
python tests/test_2_model_architecture.py
python tests/test_3_loss_functions.py
python tests/test_4_inference_pipeline.py
python tests/test_5_cross_task_features.py
```

### 📈 **테스트 결과**

```
🎯 SafeStrp 전체 테스트 결과:
   총 테스트: 5개 모듈
   성공: 5개 모듈 ✅
   실패: 0개 모듈 ✅
   성공률: 100.0% 🎉

🚀 성능 검증:
   GPU 처리량: 44+ images/sec
   메모리 효율성: 모델 64MB, 추론 25-200MB
   안정성: 완벽한 일관성 (변동계수 < 5%)
```

## 🔬 **MTPSL Cross-task Consistency**

### 💡 **핵심 아이디어**

MTPSL(Multi-Task Partial Supervised Learning)은 태스크 간 상호 보완적 정보를 활용하여 학습 성능을 향상시키는 기법입니다.

### 🔄 **동작 원리**

1. **임베딩 생성**: 각 태스크의 예측과 GT를 512차원 임베딩으로 변환
2. **일관성 측정**: Cosine similarity로 태스크 간 일관성 계산
3. **손실 통합**: Cross-task loss를 전체 손실에 통합

```python
# Surface ↔ Depth 일관성 예시
surface_pred_emb = project_surface(surface_pred)  # (B, 512)
depth_pred_emb = project_depth(depth_pred)        # (B, 512)

# 일관성 손실
consistency_loss = cosine_similarity_loss(
    surface_pred_emb, 
    depth_pred_emb
)
```

## 📁 **프로젝트 구조**

```
safestrp/
├── 📂 src/                     # 핵심 소스 코드
│   ├── 📂 core/                # 핵심 모델 및 백본
│   │   ├── model.py            # ThreeTaskDSPNet 메인 모델
│   │   └── backbone.py         # DSPNetBackbone
│   ├── 📂 heads/               # 태스크별 헤드
│   │   ├── detection.py        # SSD Detection Head
│   │   ├── segmentation.py     # Surface Segmentation Head
│   │   ├── depth.py            # Depth Regression Head
│   │   └── cross_task.py       # Cross-task Projection Head
│   ├── 📂 losses/              # 손실 함수
│   │   ├── base.py             # 기본 손실 함수들
│   │   ├── multitask.py        # UberNetMTPSLLoss
│   │   ├── task_specific.py    # 태스크별 손실
│   │   └── utils.py            # 손실 유틸리티
│   ├── 📂 utils/               # 유틸리티
│   │   ├── anchors.py          # SSD 앵커 생성
│   │   └── nms.py              # NMS 알고리즘
│   └── 📂 data/                # 데이터 처리
├── 📂 tests/                   # 테스트 시스템
│   ├── 🔬 test_1_imports_dependencies.py    # 1단계 테스트
│   ├── 🏗️ test_2_model_architecture.py     # 2단계 테스트
│   ├── 💎 test_3_loss_functions.py         # 3단계 테스트
│   ├── 🚀 test_4_inference_pipeline.py     # 4단계 테스트
│   ├── 🔗 test_5_cross_task_features.py    # 5단계 테스트
│   └── 🎮 run_all_tests.py                 # 전체 테스트 실행기
├── 📂 docs/                    # 문서
├── 📂 scripts/                 # 유틸리티 스크립트
├── 📂 configs/                 # 설정 파일
├── 📂 references/              # 참조 구현 (MTPSL, UberNet)
├── 📂 archive/                 # 아카이브된 파일들
├── 📋 requirements.txt         # 의존성 목록
└── 📖 README.md                # 프로젝트 문서
```

## 🔧 **개발 및 기여**

### 🛠️ **개발 환경 설정**

```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 코드 품질 검사
flake8 src/
black src/

# 테스트 실행
python -m pytest tests/
```

## 📚 **참조 및 인용**

이 프로젝트는 다음 연구들을 기반으로 합니다:

- **MTPSL**: Multi-Task Partial Supervised Learning
- **UberNet**: Training a Universal Convolutional Neural Network
- **DSPNet**: Driving Scene Perception Network