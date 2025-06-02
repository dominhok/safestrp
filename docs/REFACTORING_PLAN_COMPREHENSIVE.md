# SafeStrp 프로젝트 종합 리팩터링 계획서

## 🎯 리팩터링 목표

1. **파일 크기 최적화**: 300줄 이하로 각 파일 크기 제한
2. **기능별 모듈화**: 관련 기능들을 논리적으로 그룹화
3. **직관적 네이밍**: 파일명, 클래스명, 함수명 일관성 확보
4. **의존성 최소화**: 순환 참조 제거 및 명확한 의존성 구조
5. **코드 재사용성**: 공통 기능 추출 및 베이스 클래스 활용

## 📊 현재 문제점 분석

### 큰 파일들 (리팩터링 필요)
- `src/losses.py` (1498줄) → 6개 모듈로 분리
- `src/heads.py` (831줄) → 4개 모듈로 분리  
- `utils/dataset.py` (826줄) → 3개 모듈로 분리
- `src/nms.py` (425줄) → 2개 모듈로 분리
- `src/model.py` (360줄) → 적정 크기, 구조 개선만

### 불필요한 파일/디렉토리
- `MTPSL/` 디렉토리 (참조용 코드) → `references/` 로 이동
- `UberNet/` 디렉토리 (참조용 코드) → `references/` 로 이동
- 여러 테스트 파일들 → `tests/` 디렉토리로 통합
- 여러 마크다운 문서들 → `docs/` 디렉토리로 통합

## 🏗️ 새로운 프로젝트 구조

```
safestrp/
├── src/
│   ├── core/                    # 핵심 모델 구성 요소
│   │   ├── __init__.py
│   │   ├── model.py            # 메인 ThreeTaskDSPNet (< 300줄)
│   │   ├── backbone.py         # DSPNetBackbone (현재 95줄)
│   │   └── anchors.py          # SSD Anchor 생성 (현재 248줄)
│   │
│   ├── heads/                   # 태스크별 헤드들 (heads.py 831줄 분리)
│   │   ├── __init__.py
│   │   ├── base.py             # 공통 헤드 베이스 클래스
│   │   ├── detection.py        # MultiTaskDetectionHead
│   │   ├── segmentation.py     # PyramidPoolingSegmentationHead
│   │   ├── depth.py           # DepthRegressionHead
│   │   └── cross_task.py      # CrossTaskProjectionHeads
│   │
│   ├── losses/                  # 손실 함수들 (losses.py 1498줄 분리) 
│   │   ├── __init__.py
│   │   ├── base.py             # BaseLoss, ProjectionMixin
│   │   ├── focal.py            # FocalLoss, FocalSmoothL1Loss
│   │   ├── segmentation.py     # CrossEntropySegmentationLoss, DepthLoss
│   │   ├── multitask.py        # UberNetMTPSLLoss
│   │   └── utils.py            # IoU, hard negative mining 등
│   │
│   ├── data/                    # 데이터 관련 (dataset.py 826줄 분리)
│   │   ├── __init__.py
│   │   ├── base.py             # BaseDataset 클래스
│   │   ├── transforms.py       # 데이터 변환
│   │   └── loaders.py          # 데이터로더 및 collate 함수
│   │
│   ├── utils/                   # 유틸리티 함수들
│   │   ├── __init__.py
│   │   ├── nms.py              # NMS 관련 (nms.py 425줄 개선)
│   │   ├── metrics.py          # 평가 메트릭
│   │   ├── visualization.py    # 시각화 함수들
│   │   └── stereo_depth.py     # 스테레오 depth (현재 252줄)
│   │
│   └── __init__.py
│
├── configs/                     # 설정 파일들 (현재 구조 유지)
│   ├── __init__.py
│   ├── config.py
│   └── yaml_config.py
│
├── scripts/                     # 실행 스크립트들
│   ├── train.py
│   ├── test.py
│   └── train_yaml.py
│
├── tests/                       # 테스트 파일들 (새로 생성)
│   ├── __init__.py
│   ├── test_model.py           # comprehensive_system_check.py 개선
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_integration.py     # test_system_consistency.py 개선
│
├── docs/                        # 문서들 (새로 생성)
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── MTPSL_UBERNET_ANALYSIS.md
│   └── REFACTORING_PLAN.md
│
├── references/                  # 참조 코드들 (새로 생성)
│   ├── MTPSL/                  # 기존 MTPSL/ 이동
│   └── UberNet/                # 기존 UberNet/ 이동
│
├── data/                        # 데이터셋 (현재 구조 유지)
├── checkpoints/                 # 체크포인트 (현재 구조 유지)
├── runs/                        # 실험 로그 (현재 구조 유지)
├── logs/                        # 로그 (현재 구조 유지)
├── README.md
├── requirements.txt
└── .gitignore
```

## 📋 리팩터링 실행 단계

### Phase 1: 디렉토리 구조 준비 및 참조 코드 정리
1. 새 디렉토리 구조 생성
2. 참조 코드들 이동 (`MTPSL/`, `UberNet/` → `references/`)
3. 문서들 이동 (`docs/` 디렉토리로)
4. 테스트 파일들 정리 (`tests/` 디렉토리로)

### Phase 2: Core 모듈 리팩터링
1. `src/core/` 생성 및 `model.py`, `backbone.py`, `anchors.py` 이동/개선
2. 메인 모델 클래스 구조 개선

### Phase 3: Heads 모듈 분리 (heads.py 831줄)
1. `src/heads/` 생성
2. 각 헤드 클래스들을 개별 파일로 분리
3. 공통 기능 베이스 클래스 추출

### Phase 4: Losses 모듈 완성 (이미 시작됨)
1. 기존 분리 작업 완성
2. 새 구조로 이동 및 통합 테스트

### Phase 5: Data 모듈 분리 (dataset.py 826줄)
1. `src/data/` 생성
2. 데이터셋, 변환, 로더 기능 분리

### Phase 6: Utils 모듈 정리
1. NMS 관련 기능 개선
2. 새로운 유틸리티 모듈들 추가

### Phase 7: 통합 테스트 및 검증
1. 모든 import 경로 수정
2. 종합 시스템 테스트
3. 성능 및 정확도 검증

## 🔧 네이밍 컨벤션

### 파일명
- 소문자 + 언더스코어: `multi_task_loss.py`
- 기능을 명확히 표현: `detection_head.py`

### 클래스명
- PascalCase: `MultiTaskDetectionHead`
- 기능 + 타입: `FocalLoss`, `DepthEstimationHead`

### 함수명
- snake_case: `compute_detection_loss`
- 동사 + 명사: `extract_features`, `match_anchors`

## ⚠️ 주의사항

1. **점진적 리팩터링**: 한 번에 모든 것을 바꾸지 않고 단계별로 진행
2. **테스트 우선**: 각 단계마다 기능 테스트 실행
3. **백워드 호환성**: 기존 스크립트들이 동작하도록 import 별칭 제공
4. **성능 유지**: 리팩터링 후에도 동일한 성능 보장
5. **문서화**: 각 모듈별로 명확한 docstring 작성

## 🎯 예상 효과

1. **코드 가독성 향상**: 300줄 이하의 작은 파일들
2. **유지보수성 향상**: 기능별 모듈화로 수정 영향 범위 최소화
3. **재사용성 향상**: 공통 기능 베이스 클래스화
4. **테스트 용이성**: 모듈별 독립 테스트 가능
5. **확장성 향상**: 새로운 기능 추가 시 명확한 위치 제공 