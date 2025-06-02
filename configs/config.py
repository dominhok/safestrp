"""
Configuration Settings for ThreeTaskDSPNet

Detection + Surface + Depth 3태스크 멀티태스크 학습을 위한 모든 하이퍼파라미터와 설정값들을 한 곳에서 관리합니다.
"""

from dataclasses import dataclass
from typing import Tuple, List
from types import SimpleNamespace


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model parameters
    num_detection_classes: int = 29
    num_surface_classes: int = 7  # 실제 XML 분석 결과: 6개 주요 라벨 + background
    input_size: Tuple[int, int] = (512, 512)
    pretrained_backbone: bool = True
    
    # Detection head configuration
    anchors_per_location_list: List[int] = None
    
    def __post_init__(self):
        if self.anchors_per_location_list is None:
            self.anchors_per_location_list = [4, 4, 6, 6, 6, 4, 4]  # 7-level pyramid


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training parameters
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Dataset parameters
    max_samples: int = 2000
    train_ratio: float = 0.8
    num_workers: int = 2
    
    # 3태스크 Loss weights
    detection_weight: float = 1.0
    surface_weight: float = 2.0
    depth_weight: float = 1.0  # depth_weight로 변경 (distance_weight에서)
    
    # Optimizer and scheduler
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    step_size: int = 15
    gamma: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    val_frequency: int = 1  # epochs
    save_frequency: int = 5  # epochs


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Paths
    base_dir: str = "data/original_dataset"
    checkpoint_dir: str = "checkpoints/two_task"
    log_dir: str = "logs/two_task_training"
    
    # Preprocessing
    input_size: Tuple[int, int] = (512, 512)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data augmentation (for future enhancement)
    use_augmentation: bool = False
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 10
    color_jitter_prob: float = 0.3


@dataclass
class SystemConfig:
    """System configuration."""
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = True
    
    # Logging
    log_level: str = "INFO"
    tensorboard_log: bool = True
    print_frequency: int = 10  # batches
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Model saving
    save_best_only: bool = True
    save_last: bool = True
    
    # Memory optimization
    pin_memory: bool = True
    non_blocking: bool = True


class Config:
    """Main configuration class combining all configs."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__
        }
    
    def print_config(self):
        """Print current configuration."""
        print("🔧 현재 설정:")
        print("=" * 50)
        
        print("\n📊 모델 설정:")
        for key, value in self.model.__dict__.items():
            print(f"   {key}: {value}")
        
        print("\n🎯 훈련 설정:")
        for key, value in self.training.__dict__.items():
            print(f"   {key}: {value}")
        
        print("\n📁 데이터 설정:")
        for key, value in self.data.__dict__.items():
            print(f"   {key}: {value}")
        
        print("\n⚙️  시스템 설정:")
        for key, value in self.system.__dict__.items():
            print(f"   {key}: {value}")
        print()


# Pre-defined configurations for different scenarios

def get_quick_test_config() -> Config:
    """Quick test configuration with minimal parameters."""
    config = Config()
    config.training.epochs = 5
    config.training.batch_size = 2
    config.training.max_samples = 100
    config.training.val_frequency = 1
    config.training.save_frequency = 2
    return config


def get_full_training_config() -> Config:
    """Full training configuration for production."""
    config = Config()
    config.training.epochs = 100
    config.training.batch_size = 8
    config.training.max_samples = 10000
    config.training.learning_rate = 5e-5
    config.training.patience = 15
    return config


def get_debug_config() -> Config:
    """Debug configuration for development."""
    config = Config()
    config.training.epochs = 2
    config.training.batch_size = 1
    config.training.max_samples = 10
    config.training.num_workers = 0
    config.system.mixed_precision = False
    config.training.print_frequency = 1
    return config


def get_massive_dataset_config() -> Config:
    """대규모 데이터셋 훈련을 위한 설정"""
    config = Config()
    
    # Dataset settings
    config.dataset = SimpleNamespace()
    config.dataset.base_dir = "data/full_dataset"  # 통합된 데이터셋 경로
    config.dataset.batch_size = 24
    config.dataset.num_workers = 12
    config.dataset.max_samples = 5000000  # 5M samples
    config.dataset.use_depth = False  # 임시로 Depth 데이터 비활성화
    config.dataset.camera_type = "2K"  # 카메라 타입
    
    # Training settings
    config.training = SimpleNamespace()
    config.training.epochs = 30
    config.training.learning_rate = 1e-5
    config.training.weight_decay = 1e-4
    config.training.patience = 10
    config.training.use_amp = True  # Mixed Precision
    config.training.gradient_clip = 1.0
    config.training.max_grad_norm = 1.0  # Gradient clipping 추가
    config.training.val_frequency = 1
    config.training.save_frequency = 5
    
    # Model settings
    config.model = SimpleNamespace()
    config.model.input_size = (512, 512)
    config.model.num_classes = 29  # Detection classes
    config.model.surface_classes = 7  # Surface classes
    config.model.backbone = 'resnet50'
    config.model.pretrained = True
    
    # Loss settings
    config.loss = SimpleNamespace()
    config.loss.detection_weight = 1.0
    config.loss.surface_weight = 1.0
    config.loss.depth_weight = 1.0  # Depth 손실 가중치
    config.loss.use_advanced = True  # Advanced loss with IoU matching
    config.loss.neg_pos_ratio = 3.0  # Hard negative mining ratio
    
    # Optimizer settings
    config.optimizer = SimpleNamespace()
    config.optimizer.type = 'AdamW'
    config.optimizer.lr = config.training.learning_rate
    config.optimizer.weight_decay = config.training.weight_decay
    config.optimizer.betas = (0.9, 0.999)
    
    # Scheduler settings
    config.scheduler = SimpleNamespace()
    config.scheduler.type = 'ReduceLROnPlateau'
    config.scheduler.factor = 0.5
    config.scheduler.patience = 5
    config.scheduler.min_lr = 1e-7
    
    # Save settings
    config.save = SimpleNamespace()
    config.save.checkpoint_dir = "checkpoints"
    config.save.save_every = 5  # epochs
    config.save.keep_best = True
    
    # Logging
    config.logging = SimpleNamespace()
    config.logging.log_dir = "runs"
    config.logging.log_every = 100  # batches
    config.logging.use_tensorboard = True
    
    print("🚀 대규모 데이터셋 설정 로드:")
    print(f"   데이터 경로: {config.dataset.base_dir}")
    print(f"   배치 크기: {config.dataset.batch_size}")
    print(f"   최대 샘플: {config.dataset.max_samples:,}개")
    print(f"   작업자 수: {config.dataset.num_workers}")
    print(f"   학습률: {config.training.learning_rate}")
    print(f"   에폭: {config.training.epochs}")
    print(f"   혼합 정밀도: {config.training.use_amp}")
    print(f"   Depth 사용: {config.dataset.use_depth}")
    
    return config


def get_gpu_optimized_config() -> Config:
    """GPU-optimized configuration for maximum performance."""
    config = Config()
    config.training.batch_size = 16
    config.training.num_workers = 8
    config.system.mixed_precision = True
    config.system.pin_memory = True
    config.system.non_blocking = True
    return config


if __name__ == "__main__":
    # Test configurations
    print("🧪 설정 테스트")
    
    # Default config
    config = Config()
    config.print_config()
    
    # Quick test config
    print("\n🚀 Quick Test 설정:")
    quick_config = get_quick_test_config()
    quick_config.print_config()
    
    # Dictionary update test
    print("\n🔄 Dictionary 업데이트 테스트:")
    update_dict = {
        'training': {
            'batch_size': 16,
            'learning_rate': 2e-4
        },
        'model': {
            'num_detection_classes': 30
        }
    }
    config.update_from_dict(update_dict)
    print(f"   업데이트된 batch_size: {config.training.batch_size}")
    print(f"   업데이트된 learning_rate: {config.training.learning_rate}")
    print(f"   업데이트된 detection_classes: {config.model.num_detection_classes}") 