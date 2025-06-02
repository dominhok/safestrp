"""
Utility functions for SafeStrp project.

Contains NMS, metrics, visualization, and other helper functions.
"""

from .nms import (
    single_class_nms,
    multiclass_nms,
    batch_nms,
    decode_ssd_predictions,
    bbox_iou
)

try:
    from .stereo_depth import (
        process_depth_sample,
        StereoDepthCalculator
    )
    
    # Backward compatibility aliases
    generate_stereo_depth = process_depth_sample
    calculate_disparity = process_depth_sample  # Placeholder
    
    def depth_to_disparity(*args, **kwargs):
        """Legacy function - implementation needed if used."""
        pass
    
    def disparity_to_depth(*args, **kwargs):
        """Legacy function - implementation needed if used."""
        pass
        
except ImportError as e:
    print(f"Warning: Could not import stereo_depth functions: {e}")
    # Create dummy functions
    def process_depth_sample(*args, **kwargs): pass
    def generate_stereo_depth(*args, **kwargs): pass
    def calculate_disparity(*args, **kwargs): pass
    def depth_to_disparity(*args, **kwargs): pass
    def disparity_to_depth(*args, **kwargs): pass
    class StereoDepthCalculator: pass

# Backward compatibility aliases for NMS
nms = single_class_nms
apply_nms_with_class_threshold = multiclass_nms
apply_nms_with_score_threshold = batch_nms

# Legacy aliases that might be missing
def convert_to_corner_format(*args, **kwargs):
    """Legacy function - implementation needed if used."""
    pass

def convert_to_center_format(*args, **kwargs):
    """Legacy function - implementation needed if used."""
    pass

__all__ = [
    # Core NMS functions
    'single_class_nms',
    'multiclass_nms', 
    'batch_nms',
    'decode_ssd_predictions',
    'bbox_iou',
    
    # Backward compatibility aliases
    'nms',
    'apply_nms_with_class_threshold',
    'apply_nms_with_score_threshold',
    'convert_to_corner_format',
    'convert_to_center_format',
    
    # Stereo depth functions
    'process_depth_sample',
    'StereoDepthCalculator',
    'generate_stereo_depth',
    'calculate_disparity',
    'depth_to_disparity',
    'disparity_to_depth'
] 