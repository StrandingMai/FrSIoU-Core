# FrSIoU: Fractional Semantic IoU Loss

Core loss code for *Geometry Meets Semantics: Fractional Gradient Stabilization for Semantic-Driven Bounding Box Optimization in Remote Sensing Object Detection*. FrSIoU is a training loss for oriented bounding box regression. This initial release contains the loss used in the original MMRotate experiments. The full experiment code is planned for release upon acceptance.

## Core implementation

[`fm_semantic_iou_loss.py`](fm_semantic_iou_loss.py) contains `FMSemanticIoULoss`. It combines differentiable rotated IoU with two cues from regions of interest (RoIs): mean squared feature distance within the boxes and optional cosine similarity over their surrounding context. It detaches the semantic features before computing the score and applies a lower bound to the resulting geometric loss weight. A finite history of batch-mean semantic IoU values provides a detached fractional-order scaling factor. See the manuscript's Method subsections **Semantics-Driven Bounding Box Optimization** and **Fractional IoU Loss for Gradient Stabilization**.

## Use with MMRotate

The loss expects an MMRotate environment with PyTorch and MMCV's `diff_iou_rotated_2d` operator. The original handoff was checked on Ubuntu with Python 3.7, PyTorch 1.9.0 (CUDA 11.1), `mmcv-full` 1.7.2 and MMDetection 2.28.2.

1. Copy `fm_semantic_iou_loss.py` to `mmrotate/models/losses/` in the compatible MMRotate source tree. Import `FMSemanticIoULoss` from `.fm_semantic_iou_loss` in that directory's `__init__.py` so the loss registers with MMRotate.
2. In the detector's training head, pass matched positive boxes and aligned RoI features to the loss. Supply decoded boxes in `(cx, cy, w, h, angle)` format and shape `(N, 5)`, with angles in radians. Flatten the intra-region features to `(N, C)` and supply both context tensors together if using the context term (`(N, C)` or `(N, C, H, W)`).

```python
from mmrotate.models.losses.fm_semantic_iou_loss import FMSemanticIoULoss

# Initialize once with the training head so the sIoU history persists.
loss_fn = FMSemanticIoULoss(nu=0.8, lambda_=0.3, lambda_ctx=0.5,
                            alpha=0.3, M=5, loss_weight=2.0)

# During training, with inputs on the same device as the loss module:
loss_bbox = loss_fn(pred_boxes, gt_boxes, pred_roi_features, gt_roi_features,
                    sem_pred_ctx=pred_context, sem_gt_ctx=gt_context)
```

The example values are from the archived HRSC2016 Oriented R-CNN configuration. The detector, RoI feature extraction, training and inference entry points, checkpoints, and datasets are outside the scope of this initial code release.

## License

The [Apache 2.0 license file](LICENSE) is carried over from the source MMRotate tree in the original handoff.
