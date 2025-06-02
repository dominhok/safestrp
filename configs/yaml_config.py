"""
YAML-based Configuration System for SafeStrp

깔끔하고 직관적인 YAML 기반 설정 시스템.
핵심 설정만 외부에서 컨트롤하고 나머지는 합리적 기본값 사용.
"""

import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class YAMLConfig:
    """YAML에서 로드된 설정을 담는 깔끔한 클래스."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """YAML dict를 받아서 속성으로 변환."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 중첩 dict는 recursive하게 YAMLConfig 객체로 변환
                setattr(self, key, YAMLConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리 스타일 접근 지원."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근 지원."""
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        """in 연산자 지원."""
        return hasattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """다시 딕셔너리로 변환."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, YAMLConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_yaml_config(config_path: str) -> YAMLConfig:
    """
    YAML 파일에서 설정을 로드합니다.
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        YAMLConfig 객체
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return YAMLConfig(config_dict)


def get_default_config() -> YAMLConfig:
    """기본 설정을 반환합니다 (YAML 파일이 없을 때)."""
    default_dict = {
        'model': {
            'num_detection_classes': 29,
            'num_surface_classes': 7,
            'input_size': [512, 512],
            'pretrained_backbone': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'loss_weights': {
                'detection': 1.0,
                'surface': 1.0,
                'depth': 0.5
            },
            'optimizer': 'AdamW',
            'scheduler': 'ReduceLROnPlateau',
            'patience': 10,
            'max_grad_norm': 1.0,
            'mixed_precision': True
        },
        'dataset': {
            'base_dir': 'data/original_dataset',
            'max_samples': 2000,
            'val_split': 0.2
        },
        'system': {
            'num_workers': 4,
            'pin_memory': False,
            'non_blocking': True,
            'device': 'auto'
        },
        'checkpoint': {
            'save_dir': 'checkpoints',
            'save_interval': 10,
            'keep_best': True
        },
        'logging': {
            'log_dir': 'logs',
            'tensorboard': True,
            'print_interval': 50
        }
    }
    
    return YAMLConfig(default_dict)


def load_training_config(config_path: str = None) -> YAMLConfig:
    """
    훈련용 설정을 로드합니다.
    
    Args:
        config_path: YAML 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        YAMLConfig 객체
    """
    if config_path is None:
        # 기본 경로들을 순서대로 시도
        default_paths = [
            'configs/train_config.yaml',
            'train_config.yaml',
            'config.yaml'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            print("⚠️  YAML config 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return get_default_config()
    
    try:
        config = load_yaml_config(config_path)
        print(f"✅ Config 로드 완료: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Config 로드 실패: {e}")
        print("기본 설정을 사용합니다.")
        return get_default_config()


def save_config(config: YAMLConfig, save_path: str):
    """설정을 YAML 파일로 저장합니다."""
    config_dict = config.to_dict()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Config 저장 완료: {save_path}")


# 편의 함수들
def quick_config(
    batch_size: int = 8,
    learning_rate: float = 0.001,
    epochs: int = 100,
    **kwargs
) -> YAMLConfig:
    """빠른 설정 생성."""
    config = get_default_config()
    
    # 주요 설정 업데이트
    config.training.batch_size = batch_size
    config.training.learning_rate = learning_rate
    config.training.epochs = epochs
    
    # 추가 설정 업데이트
    for key, value in kwargs.items():
        if '.' in key:
            # 중첩 키 처리 (예: "training.weight_decay")
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)
    
    return config 