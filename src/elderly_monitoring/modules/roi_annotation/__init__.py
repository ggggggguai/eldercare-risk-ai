from elderly_monitoring.modules.roi_annotation.service import RoiAnnotationError, annotate_roi_image
from elderly_monitoring.modules.roi_annotation.validation import RoiValidationError, validate_roi_payload

__all__ = [
    "RoiAnnotationError",
    "RoiValidationError",
    "annotate_roi_image",
    "validate_roi_payload",
]
