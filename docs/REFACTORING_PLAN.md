# SafeStrp 리팩토링 계획

## 📋 현재 상태 분석

### ✅ 완료된 구현
- **UberNet + MTPSL 하이브리드 시스템** 완전 구현
- **3태스크 멀티태스크 학습** (Detection + Surface + Depth)
- **Partial dataset 지원** (Surface: 46K, Depth: 15K 샘플)
- **Cross-task consistency** (Seg ↔ Depth)
- **모든 테스트 통과** (5/5)

### 🔍 발견된 개선점
1. **코드 일관성**: torch.nn.functional 임포트 방식 통일 필요
2. **디렉토리 정리**: UberNet/, MTPSL/ 레거시 폴더 정리
3. **모듈 구조**: 일부 중복 코드 및 불필요한 imports 존재
4. **성능 최적화**: Anchor 생성 및 Loss 계산 최적화 가능

## 🎯 리팩토링 목표

1. **코드 품질 향상**: 일관성, 가독성, 유지보수성
2. **성능 최적화**: 메모리 사용량 및 연산 속도 개선
3. **모듈화 강화**: 더 명확한 역할 분리 및 의존성 관리
4. **확장성 증대**: 새로운 태스크 추가 용이성

## 🗂️ 리팩토링 단계별 계획

### Phase 1: 코드 정리 및 일관성 개선
**목표**: 코드 품질 및 일관성 향상

#### 1.1 Import 및 코딩 스타일 통일
```python
# 현재 문제점
- torch.nn.functional as F 일관성 없음
- Type hints 부분적 적용
- Docstring 스타일 혼재

# 개선 방안
- torch import 방식 통일
- 모든 함수에 type hints 추가
- Google style docstring 통일
```

#### 1.2 불필요한 파일 및 디렉토리 정리
```
제거 대상:
├── UberNet/ (레거시)
├── MTPSL/ (레거시)  
├── test_*.py (완료된 테스트 파일들)
└── __pycache__/ (캐시 파일들)

정리 후 구조:
├── src/ (핵심 구현)
├── utils/ (유틸리티)
├── configs/ (설정)
├── scripts/ (실행 스크립트)
└── docs/ (문서)
```

#### 1.3 중복 코드 제거
```python
# 중복 제거 대상
- Loss 함수들의 공통 로직
- Dataset loading 중복 코드
- Anchor generation 최적화
```

### Phase 2: 아키텍처 최적화
**목표**: 성능 및 메모리 효율성 개선

#### 2.1 모델 아키텍처 최적화
```python
# 최적화 대상
class ThreeTaskDSPNet:
    - Anchor 생성을 모델 외부로 분리
    - Cross-task projection 조건부 활성화
    - Memory-efficient forward pass
    
# 개선 방안
- Lazy anchor generation
- Dynamic computation graph
- Gradient checkpointing 옵션
```

#### 2.2 Loss 함수 최적화
```python
# 현재 UberNetMTPSLLoss 최적화
- 불필요한 계산 제거
- Batch-wise 병렬 처리
- Memory-efficient cross-task loss

# 추가 개선
- Adaptive loss weighting
- Loss balancing 자동화
```

#### 2.3 데이터 로딩 최적화
```python
# Dataset 최적화
- Memory-mapped file loading
- Efficient batch collation
- Multi-process data loading
- Cache mechanism for frequent samples
```

### Phase 3: 모듈화 및 확장성 강화
**목표**: 유지보수성 및 확장성 향상

#### 3.1 모듈 구조 재설계
```python
# 새로운 모듈 구조
src/
├── core/           # 핵심 아키텍처
│   ├── model.py    # 메인 모델
│   ├── backbone.py # 백본 네트워크
│   └── heads/      # 태스크별 헤드들
├── training/       # 학습 관련
│   ├── losses.py   # Loss 함수들
│   ├── trainer.py  # 학습 루프
│   └── metrics.py  # 평가 지표
├── data/           # 데이터 처리
│   ├── datasets.py # 데이터셋
│   ├── transforms.py # 데이터 변환
│   └── loaders.py  # 데이터 로더
└── utils/          # 유틸리티
    ├── anchors.py  # Anchor 생성
    ├── nms.py      # NMS 알고리즘
    └── misc.py     # 기타 유틸
```

#### 3.2 Configuration 시스템
```python
# config/model.yaml
model:
  backbone: resnet50
  num_detection_classes: 29
  num_surface_classes: 7
  enable_cross_task: true
  
training:
  batch_size: 8
  learning_rate: 0.001
  loss_weights:
    detection: 1.0
    surface: 2.0
    depth: 1.0
    cross_task: 0.5
```

#### 3.3 Plugin 시스템
```python
# 새로운 태스크 추가 용이성
class TaskHead(ABC):
    @abstractmethod
    def forward(self, features): pass
    
    @abstractmethod  
    def compute_loss(self, pred, target): pass

# 새로운 cross-task 관계 추가
class CrossTaskConsistency(ABC):
    @abstractmethod
    def compute_consistency_loss(self, task1, task2): pass
```

### Phase 4: 성능 모니터링 및 최적화
**목표**: 실제 사용 환경에서의 최적화

#### 4.1 Profiling 및 Benchmarking
```python
# 성능 측정 시스템
- GPU memory usage tracking
- Inference time profiling  
- Training throughput monitoring
- Loss convergence analysis
```

#### 4.2 모델 경량화
```python
# 경량화 옵션
- Knowledge distillation
- Pruning for deployment
- Quantization support
- Mobile/Edge optimization
```

#### 4.3 분산 학습 지원
```python
# 확장성 개선
- Multi-GPU training (DataParallel)
- Distributed training (DistributedDataParallel)
- Mixed precision training
- Gradient accumulation
```

## 📊 예상 개선 효과

### 성능 개선
- **메모리 사용량**: 20-30% 감소 예상
- **학습 속도**: 15-25% 향상 예상  
- **추론 속도**: 10-20% 향상 예상

### 개발 효율성
- **코드 가독성**: 크게 향상
- **유지보수성**: 모듈화로 개선
- **확장성**: 새로운 태스크 추가 용이
- **테스트 용이성**: 단위 테스트 가능

### 실용성
- **배포 용이성**: Docker, config 기반
- **모니터링**: 학습 과정 가시화
- **디버깅**: 더 명확한 에러 메시지

## 🚀 실행 순서

1. **Week 1**: Phase 1 (코드 정리)
2. **Week 2**: Phase 2 (아키텍처 최적화)  
3. **Week 3**: Phase 3 (모듈화)
4. **Week 4**: Phase 4 (성능 최적화)
5. **Week 5**: 통합 테스트 및 문서화

## 📝 주의사항

- **기능 유지**: 현재 작동하는 모든 기능 보존
- **하위 호환성**: 기존 API 최대한 유지
- **점진적 개선**: 한 번에 모든 것을 바꾸지 않고 단계적 개선
- **테스트 우선**: 각 단계마다 충분한 테스트 수행

이 리팩토링을 통해 SafeStrp 시스템이 더욱 견고하고 확장 가능한 멀티태스크 학습 플랫폼으로 발전할 것입니다. 