"""
Propagation module - Track objects across frames using embeddings + SAM.

Supports two propagation modes:
1. Peak-based (default): Find similarity peaks, prompt SAM at those locations
2. Dense (legacy-style): Use dense patch correspondence to propagate mask probabilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Literal, TYPE_CHECKING
from dataclasses import asdict, dataclass, field
import logging

if TYPE_CHECKING:
    from ml.embed import EmbedService
    from ml.segment import SegmentService

logger = logging.getLogger(__name__)


@dataclass
class BboxCandidate:
    """A candidate bounding box with score."""

    bbox_xyxy: list[float]
    score: float
    source: str = "propagated"


# ==================== Tracking uncertainty ====================
#
# A propagation step produces one of three outcomes:
#   - "ok":           best candidate looks consistent with the source object
#   - "needs_review": a plausible candidate exists but some signal is off
#                     (size change, centre jump, low similarity, ...). The
#                     result is still returned so tracking can continue.
#   - lost:           no plausible candidate at all -> PropagationNotFoundError
#
# TrackingSignals / PropagationResult.uncertainty() are plain, JSON-friendly
# structures so a backend tracking job can emit them directly as events.

STATUS_OK = "ok"
STATUS_NEEDS_REVIEW = "needs_review"

# Default thresholds shared by backend API and frontend (keep in sync with
# frontend/src/stores/uiStore.ts).
DEFAULT_SIZE_MIN_RATIO = 0.5
DEFAULT_SIZE_MAX_RATIO = 2.0
DEFAULT_MIN_SIMILARITY = 0.5


@dataclass
class TrackingThresholds:
    """Thresholds for classifying a propagation result.

    Size ratios are relative to the previous frame and are multiplied by the
    caller's ``expected_area_ratio`` (adaptive area trend).
    """

    size_min_ratio: float = DEFAULT_SIZE_MIN_RATIO
    size_max_ratio: float = DEFAULT_SIZE_MAX_RATIO
    # Against the trusted anchor (nearest approved annotation); wider band
    # because the object may legitimately drift in size over many frames.
    anchor_min_ratio: float = 0.25
    anchor_max_ratio: float = 4.0
    # Below this on every candidate -> lost.
    lost_similarity: float = DEFAULT_MIN_SIMILARITY
    # Candidate mask smaller than this fraction of the source -> not plausible
    # (object essentially left the frame / vanished).
    lost_min_area_ratio: float = 0.05
    # Plausible but below these -> needs_review.
    uncertain_similarity: float = 0.6
    min_sam_score: float = 0.5
    # Centre displacement in units of the source bbox diagonal.
    max_center_jump: float = 1.0
    # Minimum IoU against dense prediction (advanced mode).
    min_dense_iou: float = 0.3


@dataclass
class TrackingSignals:
    """Raw per-step tracking signals (all ratios are target / reference)."""

    embed_similarity: float
    sam_score: float
    bbox_area_ratio: float  # vs previous-frame bbox
    mask_area_ratio: Optional[float]  # vs previous-frame mask (None if no source mask)
    anchor_area_ratio: Optional[float] = None  # vs nearest approved anchor mask
    center_jump: float = 0.0  # centre shift / source bbox diagonal
    expected_area_ratio: float = 1.0  # adaptive trend used for the size check
    touches_border: bool = False
    dense_iou: Optional[float] = None

    @property
    def size_ratio(self) -> float:
        """Ratio used for the size check: mask area if available, else bbox."""
        return (
            self.mask_area_ratio
            if self.mask_area_ratio is not None
            else self.bbox_area_ratio
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (float, np.floating)):
                d[k] = round(float(v), 4)
            elif isinstance(v, np.bool_):
                d[k] = bool(v)
        return d


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def compute_signals(
    source_bbox: list[float],
    source_mask_area: Optional[float],
    target_bbox: list[float],
    target_mask: Optional[np.ndarray],
    embed_similarity: float,
    sam_score: float,
    img_w: int,
    img_h: int,
    anchor_mask_area: Optional[float] = None,
    expected_area_ratio: float = 1.0,
    dense_iou: Optional[float] = None,
) -> TrackingSignals:
    """Compute raw tracking signals for one candidate."""
    src_area = _bbox_area(source_bbox)
    bbox_ratio = _bbox_area(target_bbox) / src_area if src_area > 0 else 1.0

    tgt_mask_area = (
        float(np.count_nonzero(target_mask)) if target_mask is not None else None
    )
    mask_ratio = None
    if source_mask_area and tgt_mask_area is not None:
        mask_ratio = tgt_mask_area / source_mask_area
    anchor_ratio = None
    if anchor_mask_area and tgt_mask_area is not None:
        anchor_ratio = tgt_mask_area / anchor_mask_area

    scx = (source_bbox[0] + source_bbox[2]) / 2
    scy = (source_bbox[1] + source_bbox[3]) / 2
    tcx = (target_bbox[0] + target_bbox[2]) / 2
    tcy = (target_bbox[1] + target_bbox[3]) / 2
    diag = float(
        np.hypot(source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1])
    )
    jump = float(np.hypot(tcx - scx, tcy - scy)) / diag if diag > 0 else 0.0

    margin = 1.0
    touches = (
        target_bbox[0] <= margin
        or target_bbox[1] <= margin
        or target_bbox[2] >= img_w - margin
        or target_bbox[3] >= img_h - margin
    )

    return TrackingSignals(
        embed_similarity=float(embed_similarity),
        sam_score=float(sam_score),
        bbox_area_ratio=float(bbox_ratio),
        mask_area_ratio=None if mask_ratio is None else float(mask_ratio),
        anchor_area_ratio=None if anchor_ratio is None else float(anchor_ratio),
        center_jump=jump,
        expected_area_ratio=float(expected_area_ratio),
        touches_border=bool(touches),
        dense_iou=None if dense_iou is None else float(dense_iou),
    )


def is_plausible(signals: TrackingSignals, thresholds: TrackingThresholds) -> bool:
    """Could this candidate be the tracked object at all? (else: lost)"""
    if signals.embed_similarity < thresholds.lost_similarity:
        return False
    if signals.size_ratio < thresholds.lost_min_area_ratio:
        return False
    return True


def assess_signals(
    signals: TrackingSignals, thresholds: TrackingThresholds
) -> list[str]:
    """Return uncertainty reasons (empty list = confident)."""
    reasons: list[str] = []
    exp = signals.expected_area_ratio or 1.0
    rel = signals.size_ratio / exp
    if not (thresholds.size_min_ratio <= rel <= thresholds.size_max_ratio):
        reasons.append("size_change")
    if signals.anchor_area_ratio is not None and not (
        thresholds.anchor_min_ratio
        <= signals.anchor_area_ratio
        <= thresholds.anchor_max_ratio
    ):
        reasons.append("anchor_size_drift")
    if signals.embed_similarity < thresholds.uncertain_similarity:
        reasons.append("low_similarity")
    if signals.sam_score < thresholds.min_sam_score:
        reasons.append("low_sam_score")
    if signals.center_jump > thresholds.max_center_jump:
        reasons.append("center_jump")
    if signals.dense_iou is not None and signals.dense_iou < thresholds.min_dense_iou:
        reasons.append("low_dense_iou")
    return reasons


def update_area_trend(
    expected_area_ratio: float,
    observed_area_ratio: Optional[float],
    alpha: float = 0.3,
    clamp: tuple[float, float] = (0.5, 2.0),
) -> float:
    """Running (log-space EMA) estimate of the per-frame area change.

    Feed the observed ``signals.size_ratio`` of each accepted step; pass the
    result as ``expected_area_ratio`` for the next step of the same track.
    """
    if not observed_area_ratio or observed_area_ratio <= 0:
        return expected_area_ratio
    prev = np.log(max(expected_area_ratio, 1e-6))
    obs = np.log(observed_area_ratio)
    val = float(np.exp((1 - alpha) * prev + alpha * obs))
    return float(min(clamp[1], max(clamp[0], val)))


@dataclass
class PropagationResult:
    """Result of a propagation attempt."""

    bbox: list[float]
    mask: np.ndarray
    confidence: float
    fallback_used: bool
    area_ratio: float
    method: str  # "peak", "dense", or "iou_match"
    iou_score: Optional[float] = None  # IoU with dense prediction (if verified)
    status: str = STATUS_OK  # "ok" | "needs_review"
    reasons: list[str] = field(default_factory=list)
    signals: Optional[TrackingSignals] = None

    @property
    def uncertain(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    def uncertainty(self) -> dict:
        """JSON-friendly status/reasons/signals (event payload)."""
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "signals": self.signals.to_dict() if self.signals else None,
        }


class PropagationError(Exception):
    """Base exception for propagation failures.

    ``reason`` is a short machine-readable code and ``signals`` holds the
    best (rejected) candidate's signals, if any, for event emission.
    """

    def __init__(
        self,
        message: str = "",
        reason: str = "error",
        signals: Optional[TrackingSignals] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.signals = signals

    def to_dict(self) -> dict:
        return {
            "status": "lost",
            "reason": self.reason,
            "message": str(self),
            "signals": self.signals.to_dict() if self.signals else None,
        }


class PropagationSizeMismatchError(PropagationError):
    """Object size changed too much (only raised with stop_on_size_mismatch=True)."""

    pass


class PropagationNotFoundError(PropagationError):
    """Could not find a plausible object candidate in target image (lost)."""

    pass


def generate_candidates(
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    search_expand: float = 2.0,
    num_scales: int = 5,
    num_positions: int = 9,
    aspect_ratio_tolerance: float = 0.2,
) -> list[list[float]]:
    """
    Generate candidate bounding boxes around the previous bbox location.

    Uses a multi-scale sliding window approach within an expanded search region.

    Args:
        prev_bbox: Previous bbox [x1, y1, x2, y2]
        img_w, img_h: Image dimensions
        search_expand: Factor to expand search region (2.0 = 2x the bbox size)
        num_scales: Number of scale variations to try
        num_positions: Number of position variations (3x3 grid = 9)
        aspect_ratio_tolerance: How much to vary aspect ratio

    Returns:
        List of candidate bboxes [[x1, y1, x2, y2], ...]
    """
    x1, y1, x2, y2 = prev_bbox
    prev_w = x2 - x1
    prev_h = y2 - y1
    prev_cx = (x1 + x2) / 2
    prev_cy = (y1 + y2) / 2

    # Define search region (expanded around previous bbox)
    search_w = prev_w * search_expand
    search_h = prev_h * search_expand

    # Scale factors to try (centered around 1.0)
    scale_range = 0.3  # ±30%
    scales = np.linspace(1.0 - scale_range, 1.0 + scale_range, num_scales)

    # Position offsets (grid within search region)
    grid_size = int(np.sqrt(num_positions))
    offset_range_x = search_w / 4  # Don't go too far
    offset_range_y = search_h / 4

    if grid_size > 1:
        offsets_x = np.linspace(-offset_range_x, offset_range_x, grid_size)
        offsets_y = np.linspace(-offset_range_y, offset_range_y, grid_size)
    else:
        offsets_x = [0]
        offsets_y = [0]

    candidates = []

    for scale in scales:
        for off_x in offsets_x:
            for off_y in offsets_y:
                # New center
                cx = prev_cx + off_x
                cy = prev_cy + off_y

                # New size
                w = prev_w * scale
                h = prev_h * scale

                # Create bbox
                new_x1 = cx - w / 2
                new_y1 = cy - h / 2
                new_x2 = cx + w / 2
                new_y2 = cy + h / 2

                # Clamp to image bounds
                new_x1 = max(0, min(new_x1, img_w - 1))
                new_y1 = max(0, min(new_y1, img_h - 1))
                new_x2 = max(new_x1 + 1, min(new_x2, img_w))
                new_y2 = max(new_y1 + 1, min(new_y2, img_h))

                # Skip if too small
                if (new_x2 - new_x1) < 10 or (new_y2 - new_y1) < 10:
                    continue

                candidates.append([new_x1, new_y1, new_x2, new_y2])

    # Remove duplicates (approximately)
    unique_candidates = []
    for cand in candidates:
        is_dup = False
        for existing in unique_candidates:
            if _bbox_iou(cand, existing) > 0.9:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)

    return unique_candidates


def _bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _translate_bbox_hint(
    source_bbox: list[float],
    center_x: float,
    center_y: float,
    img_w: int,
    img_h: int,
    scale: float = 1.15,
) -> list[float]:
    """Build a padded square hint around a new center point.

    Using the larger source bbox side makes the hint much more tolerant to
    object rotation than reusing the exact previous width/height.
    """
    side = max(
        1.0,
        max(source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1]) * scale,
    )

    x1 = center_x - side / 2
    y1 = center_y - side / 2
    x2 = center_x + side / 2
    y2 = center_y + side / 2

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)

    if x2 <= x1:
        x2 = min(float(img_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(img_h), y1 + 1.0)

    return [x1, y1, x2, y2]


def score_candidates(
    candidates: list[list[float]],
    prev_descriptor: np.ndarray,
    target_features: "torch.Tensor",
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    embed_service: "EmbedService",
    spatial_weight: float = 0.1,
) -> list[BboxCandidate]:
    """
    Score candidate bboxes using embedding similarity.

    Args:
        candidates: List of candidate bboxes
        prev_descriptor: Descriptor of the object from previous frame
        target_features: Feature map of target image
        prev_bbox: Previous bbox (for spatial prior)
        img_w, img_h: Image dimensions
        embed_service: EmbedService instance
        spatial_weight: Weight for spatial distance penalty

    Returns:
        List of BboxCandidate sorted by score (highest first)
    """
    num_patches_h = embed_service._img_size // embed_service._patch_size
    num_patches_w = embed_service._img_size // embed_service._patch_size

    # Reshape features to spatial
    feat_map = target_features.reshape(1, num_patches_h, num_patches_w, -1)
    feat_map = feat_map.permute(0, 3, 1, 2)  # (1, embed_dim, H, W)

    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
    max_dist = np.sqrt(img_w**2 + img_h**2)

    scored_candidates = []

    for bbox in candidates:
        # Convert bbox to feature coordinates
        x1, y1, x2, y2 = bbox
        fx1 = int(x1 / img_w * num_patches_w)
        fy1 = int(y1 / img_h * num_patches_h)
        fx2 = int(np.ceil(x2 / img_w * num_patches_w))
        fy2 = int(np.ceil(y2 / img_h * num_patches_h))

        # Clamp
        fx1 = max(0, min(fx1, num_patches_w - 1))
        fy1 = max(0, min(fy1, num_patches_h - 1))
        fx2 = max(fx1 + 1, min(fx2, num_patches_w))
        fy2 = max(fy1 + 1, min(fy2, num_patches_h))

        # Extract and pool region features
        region_feats = feat_map[:, :, fy1:fy2, fx1:fx2]
        region_desc = region_feats.mean(dim=(2, 3))
        region_desc = F.normalize(region_desc, dim=-1)
        region_desc = region_desc.squeeze(0).cpu().numpy()

        # Compute similarity
        similarity = float(np.dot(prev_descriptor, region_desc))

        # Spatial prior (penalize distance from previous location)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        spatial_penalty = spatial_weight * (dist / max_dist)

        # Combined score
        score = similarity - spatial_penalty

        scored_candidates.append(
            BboxCandidate(bbox_xyxy=bbox, score=score, source="propagated")
        )

    # Sort by score (highest first)
    scored_candidates.sort(key=lambda x: x.score, reverse=True)

    return scored_candidates


def propose_bboxes(
    prev_descriptor: np.ndarray,
    target_features: "torch.Tensor",
    prev_bbox: list[float],
    img_w: int,
    img_h: int,
    embed_service: "EmbedService",
    top_k: int = 5,
) -> list[BboxCandidate]:
    """
    Propose top-k bboxes for an object in a new frame.

    Args:
        prev_descriptor: Object descriptor from previous frame
        target_features: Feature map of target frame
        prev_bbox: Bbox from previous frame
        img_w, img_h: Target image dimensions
        embed_service: EmbedService instance
        top_k: Number of candidates to return

    Returns:
        Top-k BboxCandidates sorted by score
    """
    # Generate candidates
    candidates = generate_candidates(prev_bbox, img_w, img_h)

    if not candidates:
        return []

    # Score candidates
    scored = score_candidates(
        candidates,
        prev_descriptor,
        target_features,
        prev_bbox,
        img_w,
        img_h,
        embed_service,
    )

    return scored[:top_k]


class PropagateService:
    """
    Service for propagating annotations across frames.

    Combines:
    - DINOv3 embeddings for object matching
    - Candidate bbox generation
    - SAM for mask refinement

    Supports two modes:
    - "peak" (default): Find similarity peaks, prompt SAM
    - "dense": Legacy-style dense correspondence for mask propagation
    """

    def __init__(
        self,
        embed_service: "EmbedService",
        segment_service: "SegmentService",
    ):
        """
        Initialize propagation service.

        Args:
            embed_service: EmbedService for feature extraction
            segment_service: SegmentService for mask generation
        """
        self.embed_service = embed_service
        self.segment_service = segment_service

    def unload_model(self):
        """Clear references (actual unload happens in embed/segment services)."""
        self.embed_service = None
        self.segment_service = None

    def _rerun_segmentation_with_bbox(
        self,
        target_image: np.ndarray,
        target_bbox: list[float],
        target_image_id: Optional[str] = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Mirror the manual 'S' flow by re-segmenting from the final bbox."""
        self.segment_service.set_image(target_image, target_image_id)
        return self.segment_service.segment_with_bbox(bbox_xyxy=target_bbox)

    def _postprocess_tracked_mask(
        self,
        mask: np.ndarray,
        bbox: list[float],
        prune_thin_artifacts: bool,
    ) -> tuple[np.ndarray, list[float]]:
        """Apply conservative tracking-only mask cleanup and recompute bbox."""
        if not prune_thin_artifacts:
            return mask, bbox

        from core.masks import (
            mask_to_bbox,
            prune_thin_artifacts as prune_mask_artifacts,
        )

        cleaned_mask = prune_mask_artifacts(mask)
        cleaned_bbox = mask_to_bbox(cleaned_mask)

        if cleaned_bbox == [0, 0, 0, 0]:
            return mask, bbox

        return cleaned_mask, cleaned_bbox

    def propagate_dense(
        self,
        source_mask: np.ndarray,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        top_k: int = 5,
        temperature: float = 0.2,
    ) -> np.ndarray:
        """
        Propagate mask using dense feature correspondence (legacy DINO approach).

        This computes patch-to-patch similarities and uses them to propagate
        mask probabilities from source to target, producing a coarse mask
        prediction without requiring SAM.

        Args:
            source_mask: Binary mask from source image (H, W)
            source_features: Source image features (1, num_patches, embed_dim)
            target_features: Target image features (1, num_patches, embed_dim)
            top_k: Number of top matches to consider per target patch
            temperature: Softmax temperature (lower = sharper matching)

        Returns:
            Probability map for target image (num_patches_h, num_patches_w)
        """
        # Get patch dimensions
        num_patches = self.embed_service._img_size // self.embed_service._patch_size
        embed_dim = source_features.shape[-1]

        # Reshape features to spatial layout: (H, W, D)
        source_feats = source_features.squeeze(0).reshape(
            num_patches, num_patches, embed_dim
        )
        target_feats = target_features.squeeze(0).reshape(
            num_patches, num_patches, embed_dim
        )

        # Resize mask to patch resolution
        source_mask_resized = torch.from_numpy(source_mask).float()
        source_mask_resized = F.interpolate(
            source_mask_resized.unsqueeze(0).unsqueeze(0),
            size=(num_patches, num_patches),
            mode="nearest",
        ).squeeze()

        # Create one-hot encoding (background=0, foreground=1)
        # Shape: (H, W, 2) where [..., 0] = background prob, [..., 1] = foreground prob
        source_probs = torch.zeros(
            num_patches, num_patches, 2, device=source_feats.device
        )
        source_probs[..., 0] = 1 - source_mask_resized.to(source_feats.device)
        source_probs[..., 1] = source_mask_resized.to(source_feats.device)

        # Compute dot product similarity between all target and source patches
        # target_feats: (Ht, Wt, D), source_feats: (Hs, Ws, D)
        # Result: (Ht, Wt, Hs, Ws) - similarity of each target patch to each source patch
        dot = torch.einsum("ijd,uvd->ijuv", target_feats, source_feats)

        # Flatten spatial dimensions for top-k selection
        # Shape: (Ht*Wt, Hs*Ws)
        dot_flat = dot.flatten(0, 1).flatten(1, 2)

        # Top-k filtering: keep only top-k matches per target patch
        k_th_values = torch.topk(dot_flat, dim=1, k=top_k).values[:, -1:]
        dot_flat = torch.where(
            dot_flat >= k_th_values,
            dot_flat,
            torch.tensor(-torch.inf, device=dot_flat.device),
        )

        # Softmax to get attention weights
        weights = F.softmax(dot_flat / temperature, dim=1)

        # Propagate mask probabilities: weighted sum of source probs
        # weights: (Ht*Wt, Hs*Ws), source_probs: (Hs, Ws, 2) -> (Hs*Ws, 2)
        source_probs_flat = source_probs.flatten(0, 1)  # (Hs*Ws, 2)
        target_probs = torch.mm(weights, source_probs_flat)  # (Ht*Wt, 2)

        # Normalize probabilities
        target_probs = target_probs / (target_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Reshape to spatial and extract foreground probability
        target_probs = target_probs.reshape(num_patches, num_patches, 2)
        foreground_probs = target_probs[..., 1].cpu().numpy()

        return foreground_probs

    def propagate_with_dense_fallback(
        self,
        source_image: np.ndarray,
        source_bbox: list[float],
        source_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        source_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        mode: Literal["peak", "dense", "auto"] = "auto",
        iou_verify: bool = True,
        iou_threshold: float = 0.3,
        use_cached_masks: bool = False,
        use_bbox_hint: bool = True,
        bbox_hint_scale: float = 1.15,
        prune_thin_artifacts: bool = True,
        **kwargs,
    ) -> Optional[PropagationResult]:
        """
        Propagate annotation with optional dense fallback and IoU verification.

        Modes:
        - "peak": Use peak-based propagation only (current default behavior)
        - "dense": Use dense propagation to create coarse mask, then refine with SAM
        - "auto": Try peak first, fall back to dense if it fails

        IoU verification (when enabled):
        - Computes dense propagation mask as "ground truth"
        - Verifies SAM result has sufficient IoU with dense prediction
        - Rejects results with low IoU

        Args:
            source_image: Source image (RGB numpy array)
            source_bbox: Source bbox [x1, y1, x2, y2]
            source_mask: Source mask (required for dense mode)
            target_image: Target image (RGB numpy array)
            source_image_id: ID for caching source features
            target_image_id: ID for caching target features
            annotation_id: Annotation ID for caching
            mode: Propagation mode ("peak", "dense", or "auto")
            iou_verify: Whether to verify results against dense prediction
            iou_threshold: Minimum IoU with dense prediction to accept
            use_cached_masks: Whether to use/check SAM everything cache
            **kwargs: Additional arguments passed to propagate_annotation

        Returns:
            PropagationResult or None if propagation fails
        """
        from core.masks import mask_iou, mask_to_bbox

        img_h, img_w = target_image.shape[:2]

        # Get features for both images
        source_features = self.embed_service.get_image_features(
            source_image, image_id=source_image_id
        )
        target_features = self.embed_service.get_image_features(
            target_image, image_id=target_image_id
        )

        # Compute dense mask prediction if needed for dense mode or IoU verification
        dense_mask = None
        if source_mask is not None and (mode in ["dense", "auto"] or iou_verify):
            dense_probs = self.propagate_dense(
                source_mask, source_features, target_features
            )
            # Upsample to image resolution
            dense_probs_tensor = torch.from_numpy(dense_probs).unsqueeze(0).unsqueeze(0)
            dense_probs_upsampled = (
                F.interpolate(
                    dense_probs_tensor,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            dense_mask = (dense_probs_upsampled > 0.5).astype(np.uint8)

            logger.debug(f"Dense mask area: {dense_mask.sum()}")

        thresholds: Optional[TrackingThresholds] = kwargs.pop("thresholds", None)
        size_min_ratio = kwargs.pop("size_min_ratio", DEFAULT_SIZE_MIN_RATIO)
        size_max_ratio = kwargs.pop("size_max_ratio", DEFAULT_SIZE_MAX_RATIO)
        min_score = kwargs.pop("min_score", DEFAULT_MIN_SIMILARITY)
        if thresholds is None:
            thresholds = TrackingThresholds(
                size_min_ratio=size_min_ratio,
                size_max_ratio=size_max_ratio,
                lost_similarity=min_score,
                min_dense_iou=iou_threshold,
            )
        expected_area_ratio = kwargs.get("expected_area_ratio", 1.0)
        anchor_mask_area = kwargs.get("anchor_mask_area")
        source_mask_area = (
            float(np.count_nonzero(source_mask)) if source_mask is not None else None
        ) or None

        result: Optional[PropagationResult] = None
        # Peak result that failed dense-IoU verification: kept as a
        # needs_review fallback instead of declaring the object lost.
        unverified_peak: Optional[PropagationResult] = None
        lost_error: Optional[PropagationError] = None

        # Try peak-based propagation first (unless mode is "dense")
        if mode in ["peak", "auto"]:
            try:
                peak_result = self.propagate_annotation(
                    source_image=source_image,
                    source_bbox=source_bbox,
                    source_mask=source_mask,
                    target_image=target_image,
                    source_image_id=source_image_id,
                    target_image_id=target_image_id,
                    annotation_id=annotation_id,
                    use_bbox_hint=use_bbox_hint,
                    bbox_hint_scale=bbox_hint_scale,
                    prune_thin_artifacts=prune_thin_artifacts,
                    thresholds=thresholds,
                    **kwargs,
                )
                if iou_verify and dense_mask is not None:
                    iou_score = mask_iou(peak_result.mask, dense_mask)
                    peak_result.iou_score = iou_score
                    if peak_result.signals is not None:
                        peak_result.signals.dense_iou = iou_score
                        peak_result.reasons = assess_signals(
                            peak_result.signals, thresholds
                        )
                        peak_result.status = (
                            STATUS_NEEDS_REVIEW if peak_result.reasons else STATUS_OK
                        )
                    if iou_score < iou_threshold:
                        logger.info(
                            f"Peak result unverified: IoU {iou_score:.3f} < {iou_threshold}"
                        )
                        unverified_peak = peak_result
                    else:
                        result = peak_result
                else:
                    result = peak_result
            except PropagationError as e:
                logger.info(f"Peak propagation failed: {e}")
                lost_error = e
                if mode == "peak":
                    raise

        # Try dense + SAM (if peak failed/unverified or mode is dense)
        if result is None and mode in ["dense", "auto"] and dense_mask is not None:
            logger.info("Attempting dense propagation with SAM mask matching...")

            rows, cols = np.where(dense_mask > 0)
            if len(rows) > 0:
                center_y = int(np.mean(rows))
                center_x = int(np.mean(cols))

                self.segment_service.set_image(target_image, target_image_id)

                try:
                    bbox_hint = None
                    if use_bbox_hint:
                        dense_bbox = mask_to_bbox(dense_mask)
                        bbox_hint = _translate_bbox_hint(
                            dense_bbox,
                            center_x,
                            center_y,
                            img_w,
                            img_h,
                            scale=bbox_hint_scale,
                        )
                    mask, sam_score, bbox = self.segment_service.segment_with_point(
                        center_x, center_y, bbox_hint=bbox_hint
                    )
                    mask, bbox = self._postprocess_tracked_mask(
                        mask, bbox, prune_thin_artifacts
                    )

                    iou_score = mask_iou(mask, dense_mask)
                    if iou_score >= iou_threshold:
                        try:
                            mask, sam_score, bbox = self._rerun_segmentation_with_bbox(
                                target_image, bbox, target_image_id
                            )
                            mask, bbox = self._postprocess_tracked_mask(
                                mask, bbox, prune_thin_artifacts
                            )
                            iou_score = mask_iou(mask, dense_mask)
                        except Exception as e:
                            logger.warning(f"Dense bbox-pass refinement failed: {e}")

                    if iou_score >= iou_threshold:
                        source_desc = self.embed_service.get_object_descriptor(
                            source_image,
                            source_bbox,
                            mask=source_mask,
                            image_id=source_image_id,
                            annotation_id=annotation_id,
                        )
                        target_desc = self.embed_service.get_object_descriptor(
                            target_image, bbox, mask=mask, image_id=target_image_id
                        )
                        embed_sim = self.embed_service.compute_similarity(
                            source_desc, target_desc
                        )
                        signals = compute_signals(
                            source_bbox=source_bbox,
                            source_mask_area=source_mask_area,
                            target_bbox=bbox,
                            target_mask=mask,
                            embed_similarity=embed_sim,
                            sam_score=sam_score,
                            img_w=img_w,
                            img_h=img_h,
                            anchor_mask_area=anchor_mask_area,
                            expected_area_ratio=expected_area_ratio,
                            dense_iou=iou_score,
                        )
                        if is_plausible(signals, thresholds):
                            reasons = assess_signals(signals, thresholds)
                            confidence = (
                                0.5 * embed_sim + 0.3 * sam_score + 0.2 * iou_score
                            )
                            result = PropagationResult(
                                bbox=bbox,
                                mask=mask,
                                confidence=float(confidence),
                                fallback_used=True,
                                area_ratio=float(signals.size_ratio),
                                method="dense",
                                iou_score=iou_score,
                                status=STATUS_NEEDS_REVIEW if reasons else STATUS_OK,
                                reasons=reasons,
                                signals=signals,
                            )
                            logger.info(
                                f"Dense propagation succeeded: IoU={iou_score:.3f}, "
                                f"sim={embed_sim:.3f}, reasons={reasons}"
                            )
                        else:
                            logger.info(
                                f"Dense result implausible: {signals.to_dict()}"
                            )
                    else:
                        logger.warning(
                            f"Dense SAM result rejected: IoU {iou_score:.3f} < {iou_threshold}"
                        )
                except Exception as e:
                    logger.warning(f"Dense SAM prompting failed: {e}")

        if result is None and unverified_peak is not None:
            # Plausible peak candidate exists but disagrees with dense
            # prediction: keep tracking, flag for review.
            result = unverified_peak

        if result:
            logger.info(
                f"Propagation succeeded using method: {result.method} ({result.status})"
            )
            return result

        if lost_error is not None:
            raise PropagationNotFoundError(
                f"All propagation methods failed: {lost_error}",
                reason=lost_error.reason,
                signals=lost_error.signals,
            )
        raise PropagationNotFoundError(
            "All propagation methods failed", reason="no_candidate"
        )

    def propagate_annotation(
        self,
        source_image: np.ndarray,
        source_bbox: list[float],
        source_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        source_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        use_bbox_hint: bool = True,
        bbox_hint_scale: float = 1.15,
        prune_thin_artifacts: bool = True,
        top_k: int = 5,
        min_score: float = DEFAULT_MIN_SIMILARITY,
        search_expansion: float = 0.3,
        size_min_ratio: float = DEFAULT_SIZE_MIN_RATIO,
        size_max_ratio: float = DEFAULT_SIZE_MAX_RATIO,
        stop_on_size_mismatch: bool = False,
        expected_area_ratio: float = 1.0,
        anchor_mask_area: Optional[float] = None,
        thresholds: Optional[TrackingThresholds] = None,
    ) -> PropagationResult:
        """
        Propagate an annotation from source to target image.

        Strategy:
        1. Compute similarity map between source descriptor and target features
        2. Prompt SAM at the top similarity peaks
        3. Compute tracking signals per candidate (embedding similarity, SAM
           score, mask/bbox area ratio, anchor area ratio, centre jump)
        4. Drop implausible candidates (low similarity / vanished). If none
           remain -> PropagationNotFoundError (object lost).
        5. Pick the best candidate (prefer size-consistent, then similarity),
           refine it, and mark it ``needs_review`` if any signal is off.

        Size change alone never stops tracking unless the legacy
        ``stop_on_size_mismatch=True`` is passed.

        Args:
            min_score: Lost threshold on embedding similarity
            size_min_ratio / size_max_ratio: Allowed per-frame area ratio band
                (relative to ``expected_area_ratio``) before flagging uncertain
            expected_area_ratio: Adaptive area trend (see update_area_trend)
            anchor_mask_area: Mask area of the nearest trusted (approved)
                annotation of this object, if known
            thresholds: Full threshold override (takes precedence)

        Returns:
            PropagationResult (method="peak") with status/reasons/signals.
        """
        import torch.nn.functional as F

        if thresholds is None:
            thresholds = TrackingThresholds(
                size_min_ratio=size_min_ratio,
                size_max_ratio=size_max_ratio,
                lost_similarity=min_score,
            )

        img_h, img_w = target_image.shape[:2]

        source_area = _bbox_area(source_bbox)
        source_mask_area = (
            float(np.count_nonzero(source_mask)) if source_mask is not None else None
        ) or None

        logger.info(
            f"Source bbox: {[int(x) for x in source_bbox]}, area: {source_area:.0f}, "
            f"mask_area: {source_mask_area}"
        )

        # Get source descriptor
        source_descriptor = self.embed_service.get_object_descriptor(
            source_image,
            source_bbox,
            mask=source_mask,
            image_id=source_image_id,
            annotation_id=annotation_id,
        )

        # Get target features
        target_features = self.embed_service.get_image_features(
            target_image,
            image_id=target_image_id,
        )

        # Compute similarity map
        num_patches = self.embed_service._img_size // self.embed_service._patch_size

        feats = target_features.squeeze(0)  # (num_patches, embed_dim)
        feats = F.normalize(feats, dim=-1)

        desc_tensor = torch.from_numpy(source_descriptor).to(feats.device).float()
        similarity = torch.mv(feats, desc_tensor)  # (num_patches,)
        similarity_map = (
            similarity.reshape(num_patches, num_patches).float().cpu().numpy()
        )

        flat_sim = similarity_map.flatten()
        peak_indices = np.argsort(flat_sim)[::-1]  # Descending order

        # Set target image for SAM
        self.segment_service.set_image(target_image, target_image_id)

        def make_signals(bbox, mask, sim, sam) -> TrackingSignals:
            return compute_signals(
                source_bbox=source_bbox,
                source_mask_area=source_mask_area,
                target_bbox=bbox,
                target_mask=mask,
                embed_similarity=sim,
                sam_score=sam,
                img_w=img_w,
                img_h=img_h,
                anchor_mask_area=anchor_mask_area,
                expected_area_ratio=expected_area_ratio,
            )

        tried_points: list[tuple[float, float]] = []
        # (bbox, mask, confidence, signals, reasons)
        candidates: list[
            tuple[list[float], np.ndarray, float, TrackingSignals, list[str]]
        ] = []
        best_rejected: Optional[TrackingSignals] = None

        for i in range(min(top_k * 3, len(peak_indices))):  # Try more peaks
            peak_idx = peak_indices[i]
            py, px = divmod(peak_idx, num_patches)

            # Convert patch coords to pixel coords (center of patch)
            point_x = (px + 0.5) / num_patches * img_w
            point_y = (py + 0.5) / num_patches * img_h

            # Skip if too close to already tried points
            if any(
                abs(point_x - tx) < img_w * 0.05 and abs(point_y - ty) < img_h * 0.05
                for tx, ty in tried_points
            ):
                continue

            tried_points.append((point_x, point_y))
            if len(tried_points) > top_k:
                break

            try:
                bbox_hint = None
                if use_bbox_hint:
                    bbox_hint = _translate_bbox_hint(
                        source_bbox,
                        point_x,
                        point_y,
                        img_w,
                        img_h,
                        scale=bbox_hint_scale,
                    )

                mask, sam_score, refined_bbox = self.segment_service.segment_with_point(
                    point_x,
                    point_y,
                    bbox_hint=bbox_hint,
                )
                mask, refined_bbox = self._postprocess_tracked_mask(
                    mask,
                    refined_bbox,
                    prune_thin_artifacts,
                )
                if _bbox_area(refined_bbox) <= 0:
                    continue

                target_descriptor = self.embed_service.get_object_descriptor(
                    target_image,
                    refined_bbox,
                    mask=mask,
                    image_id=target_image_id,
                )
                embed_similarity = self.embed_service.compute_similarity(
                    source_descriptor, target_descriptor
                )
                confidence = 0.6 * embed_similarity + 0.4 * sam_score

                signals = make_signals(refined_bbox, mask, embed_similarity, sam_score)
                if not is_plausible(signals, thresholds):
                    if (
                        best_rejected is None
                        or signals.embed_similarity > best_rejected.embed_similarity
                    ):
                        best_rejected = signals
                    logger.info(f"  Implausible candidate: {signals.to_dict()}")
                    continue

                reasons = assess_signals(signals, thresholds)
                logger.info(
                    f"  Candidate: conf={confidence:.3f} reasons={reasons} "
                    f"signals={signals.to_dict()}"
                )
                candidates.append((refined_bbox, mask, confidence, signals, reasons))

            except Exception as e:
                logger.warning(f"SAM failed for peak point: {e}")
                continue

        if not candidates:
            if best_rejected is not None and (
                best_rejected.embed_similarity >= thresholds.lost_similarity
            ):
                reason = "object_vanished"
            else:
                reason = "low_similarity"
            logger.warning(
                f"Object lost ({reason}): {len(tried_points)} peaks tried, "
                f"no plausible candidate"
            )
            raise PropagationNotFoundError(
                f"Could not find object in target image ({reason})",
                reason=reason,
                signals=best_rejected,
            )

        # Prefer size-consistent candidates, then highest similarity.
        best = max(
            candidates,
            key=lambda c: ("size_change" not in c[4], c[3].embed_similarity),
        )
        bbox, mask, confidence, signals, reasons = best

        # Final bbox-pass refinement (mirrors manual 'S' flow)
        try:
            r_mask, r_sam, r_bbox = self._rerun_segmentation_with_bbox(
                target_image, bbox, target_image_id
            )
            r_mask, r_bbox = self._postprocess_tracked_mask(
                r_mask, r_bbox, prune_thin_artifacts
            )
            if _bbox_area(r_bbox) > 0:
                r_desc = self.embed_service.get_object_descriptor(
                    target_image, r_bbox, mask=r_mask, image_id=target_image_id
                )
                r_sim = self.embed_service.compute_similarity(source_descriptor, r_desc)
                r_signals = make_signals(r_bbox, r_mask, r_sim, r_sam)
                if is_plausible(r_signals, thresholds):
                    bbox, mask = r_bbox, r_mask
                    confidence = 0.6 * r_sim + 0.4 * r_sam
                    signals = r_signals
                    reasons = assess_signals(r_signals, thresholds)
        except Exception as e:
            logger.warning(f"Final bbox-pass refinement failed: {e}")

        if stop_on_size_mismatch and "size_change" in reasons:
            raise PropagationSizeMismatchError(
                f"Object size changed too much (ratio: {signals.size_ratio:.2f}x, "
                f"allowed: {thresholds.size_min_ratio:.2f}x-{thresholds.size_max_ratio:.2f}x)",
                reason="size_change",
                signals=signals,
            )

        status = STATUS_NEEDS_REVIEW if reasons else STATUS_OK
        logger.info(
            f"Propagation {status}: conf={confidence:.3f} reasons={reasons} "
            f"signals={signals.to_dict()}"
        )
        return PropagationResult(
            bbox=bbox,
            mask=mask,
            confidence=float(confidence),
            fallback_used="size_change" in reasons,
            area_ratio=float(signals.size_ratio),
            method="peak",
            status=status,
            reasons=reasons,
            signals=signals,
        )

    def propagate_sequence(
        self,
        images: list[tuple[np.ndarray, str]],  # List of (image, image_id)
        start_bbox: list[float],
        start_mask: Optional[np.ndarray],
        start_idx: int = 0,
        count: int = -1,  # -1 = all remaining
        annotation_id: Optional[int] = None,
        stop_threshold: float = 0.3,
        callback: Optional[callable] = None,
    ) -> list[tuple[int, list[float], np.ndarray, float]]:
        """
        Propagate annotation through a sequence of images.

        Args:
            images: List of (image_array, image_id) tuples
            start_bbox: Starting bbox
            start_mask: Starting mask (optional)
            start_idx: Index of starting image
            count: Number of frames to propagate (-1 = all)
            annotation_id: Annotation ID for logging/caching
            stop_threshold: Stop if confidence drops below this
            callback: Optional callback(idx, bbox, mask, score) for progress

        Returns:
            List of (image_idx, bbox, mask, confidence) for successful propagations
        """
        results = []

        current_bbox = start_bbox
        current_mask = start_mask
        current_idx = start_idx

        # Determine end index
        if count < 0:
            end_idx = len(images)
        else:
            end_idx = min(start_idx + count + 1, len(images))

        # Get source image
        source_image, source_image_id = images[current_idx]
        expected_area_ratio = 1.0

        for target_idx in range(current_idx + 1, end_idx):
            target_image, target_image_id = images[target_idx]

            try:
                result = self.propagate_annotation(
                    source_image=source_image,
                    source_bbox=current_bbox,
                    source_mask=current_mask,
                    target_image=target_image,
                    source_image_id=source_image_id,
                    target_image_id=target_image_id,
                    annotation_id=annotation_id,
                    expected_area_ratio=expected_area_ratio,
                )
            except PropagationError as e:
                logger.info(f"Propagation stopped at frame {target_idx}: {e}")
                break

            if result.confidence < stop_threshold:
                logger.info(
                    f"Propagation stopped at frame {target_idx}: "
                    f"confidence {result.confidence:.3f} < {stop_threshold}"
                )
                break

            results.append((target_idx, result.bbox, result.mask, result.confidence))

            if callback:
                callback(target_idx, result.bbox, result.mask, result.confidence)

            if result.signals is not None:
                expected_area_ratio = update_area_trend(
                    expected_area_ratio, result.signals.size_ratio
                )

            # Update for next iteration
            source_image = target_image
            source_image_id = target_image_id
            current_bbox = result.bbox
            current_mask = result.mask

        return results

    def find_all_instances(
        self,
        reference_image: np.ndarray,
        reference_bbox: list[float],
        reference_mask: Optional[np.ndarray],
        target_image: np.ndarray,
        reference_image_id: Optional[str] = None,
        target_image_id: Optional[str] = None,
        annotation_id: Optional[int] = None,
        min_similarity: float = 0.6,
        max_instances: int = 20,
        min_iou_between_instances: float = 0.0,
        max_iou_between_instances: float = 0.5,
        use_cached_masks: bool = True,
        size_tolerance: float = 0.5,
    ) -> list[PropagationResult]:
        """
        Find ALL instances of a class in the target image.

        Uses the reference annotation to define what the class looks like,
        then finds all similar objects in the target image.

        This is useful for:
        - Per-class auto-labeling (like legacy FastSAM everything mode)
        - Finding multiple instances of the same object type
        - Semi-automatic annotation of repeated objects

        Args:
            reference_image: Reference image with known instance
            reference_bbox: Bbox of reference instance [x1, y1, x2, y2]
            reference_mask: Mask of reference instance (optional but recommended)
            target_image: Image to search for instances
            reference_image_id: ID for caching reference features
            target_image_id: ID for caching target features
            annotation_id: Annotation ID for caching
            min_similarity: Minimum embedding similarity to consider a match
            max_instances: Maximum number of instances to return
            min_iou_between_instances: Min IoU between instances (for overlapping objects)
            max_iou_between_instances: Max IoU between instances (to avoid duplicates)
            use_cached_masks: Whether to use SAM everything cache
            size_tolerance: How much size can vary from reference (0.5 = 50%-200%)

        Returns:
            List of PropagationResult for each found instance
        """
        from core.masks import mask_iou

        img_h, img_w = target_image.shape[:2]
        results = []

        # Calculate reference area for size filtering
        # size_tolerance=0.5 means 50%-200% of reference size (multiplicative)
        ref_area = (reference_bbox[2] - reference_bbox[0]) * (
            reference_bbox[3] - reference_bbox[1]
        )
        min_area = ref_area / (1 + size_tolerance)
        max_area = ref_area * (1 + size_tolerance)

        # Get reference descriptor
        ref_descriptor = self.embed_service.get_object_descriptor(
            reference_image,
            reference_bbox,
            mask=reference_mask,
            image_id=reference_image_id,
            annotation_id=annotation_id,
        )

        # Get target features
        target_features = self.embed_service.get_image_features(
            target_image, image_id=target_image_id
        )

        # Compute similarity map
        num_patches = self.embed_service._img_size // self.embed_service._patch_size
        feats = target_features.squeeze(0)  # (num_patches^2, embed_dim)
        feats = F.normalize(feats, dim=-1)

        desc_tensor = torch.from_numpy(ref_descriptor).to(feats.device).float()
        similarity = torch.mv(feats, desc_tensor)
        similarity_map = similarity.reshape(num_patches, num_patches).cpu().numpy()

        # Set target image for SAM
        self.segment_service.set_image(target_image, target_image_id)

        # Strategy 1: Use cached masks if available
        if use_cached_masks:
            cached_masks = self.segment_service.get_cached_masks(target_image_id)
            if cached_masks is None:
                # Run segment everything to populate cache
                logger.info("Running segment_everything to find all masks...")
                cached_masks = self.segment_service.segment_everything(
                    target_image, target_image_id
                )

            if cached_masks:
                logger.info(
                    f"Evaluating {len(cached_masks)} cached masks for class matching..."
                )

                # Score each cached mask by its average similarity
                scored_masks = []
                for cached_mask in cached_masks:
                    # Check size constraint
                    if not (min_area <= cached_mask.area <= max_area):
                        continue

                    # Get mask's position in patch coords
                    bbox = cached_mask.bbox
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    px = int(cx / img_w * num_patches)
                    py = int(cy / img_h * num_patches)
                    px = min(max(px, 0), num_patches - 1)
                    py = min(max(py, 0), num_patches - 1)

                    # Get similarity at center
                    center_sim = float(similarity_map[py, px])

                    # Also compute descriptor similarity
                    mask_desc = self.embed_service.get_object_descriptor(
                        target_image,
                        bbox,
                        mask=cached_mask.mask,
                        image_id=target_image_id,
                    )
                    desc_sim = self.embed_service.compute_similarity(
                        ref_descriptor, mask_desc
                    )

                    # Combined score
                    score = 0.5 * center_sim + 0.5 * desc_sim

                    if score >= min_similarity:
                        scored_masks.append((cached_mask, score, desc_sim))

                # Sort by score (highest first)
                scored_masks.sort(key=lambda x: x[1], reverse=True)

                # Apply NMS to avoid duplicates
                for cached_mask, score, desc_sim in scored_masks:
                    if len(results) >= max_instances:
                        break

                    # Check IoU with existing results
                    is_duplicate = False
                    for existing in results:
                        iou = mask_iou(cached_mask.mask, existing.mask)
                        if iou > max_iou_between_instances:
                            is_duplicate = True
                            break
                        if iou > 0 and iou < min_iou_between_instances:
                            # Too much overlap but not enough - probably not the same instance
                            continue

                    if not is_duplicate:
                        area_ratio = (
                            cached_mask.area / ref_area if ref_area > 0 else 1.0
                        )
                        results.append(
                            PropagationResult(
                                bbox=cached_mask.bbox,
                                mask=cached_mask.mask,
                                confidence=score,
                                fallback_used=False,
                                area_ratio=area_ratio,
                                method="class_match",
                                iou_score=None,
                            )
                        )

                logger.info(f"Found {len(results)} instances via cached mask matching")

        # Strategy 2: Find peaks in similarity map and prompt SAM
        if len(results) < max_instances:
            logger.info("Finding additional instances via similarity peaks...")

            # Find peaks above threshold
            flat_sim = similarity_map.flatten()
            peak_indices = np.argsort(flat_sim)[::-1]

            tried_points = []
            for peak_idx in peak_indices:
                if len(results) >= max_instances:
                    break

                peak_sim = flat_sim[peak_idx]
                if peak_sim < min_similarity:
                    break

                py, px = divmod(peak_idx, num_patches)
                point_x = (px + 0.5) / num_patches * img_w
                point_y = (py + 0.5) / num_patches * img_h

                # Skip if too close to already tried points
                too_close = False
                for tx, ty in tried_points:
                    if (
                        abs(point_x - tx) < img_w * 0.05
                        and abs(point_y - ty) < img_h * 0.05
                    ):
                        too_close = True
                        break
                if too_close:
                    continue

                tried_points.append((point_x, point_y))

                try:
                    mask, sam_score, bbox = self.segment_service.segment_with_point(
                        point_x, point_y
                    )

                    # Check size constraint
                    mask_area = mask.sum()
                    if not (min_area <= mask_area <= max_area):
                        continue

                    # Verify with descriptor
                    mask_desc = self.embed_service.get_object_descriptor(
                        target_image, bbox, mask=mask, image_id=target_image_id
                    )
                    desc_sim = self.embed_service.compute_similarity(
                        ref_descriptor, mask_desc
                    )

                    if desc_sim < min_similarity:
                        continue

                    # Check IoU with existing results
                    is_duplicate = False
                    for existing in results:
                        iou = mask_iou(mask, existing.mask)
                        if iou > max_iou_between_instances:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        confidence = 0.5 * desc_sim + 0.5 * sam_score
                        area_ratio = mask_area / ref_area if ref_area > 0 else 1.0

                        results.append(
                            PropagationResult(
                                bbox=bbox,
                                mask=mask,
                                confidence=confidence,
                                fallback_used=False,
                                area_ratio=area_ratio,
                                method="peak_match",
                                iou_score=None,
                            )
                        )

                except Exception as e:
                    logger.debug(f"Peak point failed: {e}")
                    continue

        # Sort results by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(f"find_all_instances complete: found {len(results)} instances")
        return results
