// Types for AutoSeg

export interface Project {
  id: number;
  name: string;
  root_dir: string;
  image_count: number;
  yolo_summary?: object | null;
}

export interface ImageRecord {
  id: number;
  project_id: number;
  path: string;
  width: number;
  height: number;
  order_index: number;
}

export interface Label {
  id: number;
  project_id: number;
  name: string;
  color: string;
}

export interface Annotation {
  id: number;
  image_id: number;
  label_id: number;
  bbox: [number, number, number, number] | null;  // [x1, y1, x2, y2]
  polygon: number[] | null;
  mask_rle: object | null;
  source: 'manual' | 'propagated' | 'tracked' | 'yolo' | 'yolo_sam' | 'yolo_seg';
  confidence: number | null;
  status: 'approved' | 'pending' | 'needs_review' | 'rejected';
  track_id?: number | null;  // Object identity (tracks.id); null = untracked
  is_anchor?: boolean;  // approved annotation a track can re-initialise from
}

// Object identity fields returned by propagation endpoints (#4)
export interface TrackIdentityFields {
  track_id?: number | null;  // Track of the source annotation (result inherits it)
  possible_id_switch?: boolean;  // Result overlaps another track of the same label
  conflicting_track_id?: number | null;
  conflicting_annotation_id?: number | null;
}

export interface Track {
  id: number;
  project_id: number;
  label_id: number;
  name: string | null;
  notes: string | null;
  status: 'active' | 'lost' | 'ended';
  lost_at_image_id: number | null;
  annotation_count: number;
  anchor_count: number;
  first_image_index: number | null;
  last_image_index: number | null;
}

export type InteractionMode = 'view' | 'draw' | 'refine';

export interface SegmentResult {
  mask_rle: object;
  polygon: number[];
  score: number;
  bbox: [number, number, number, number];
}

export interface PropagateResult extends TrackIdentityFields {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  fallback_used?: boolean;  // True if size-mismatch fallback was used
  area_ratio?: number; // Ratio of new area to old area
  duplicate_skipped?: boolean;  // True if propagation was skipped due to duplicate
  duplicate_iou?: number;  // IoU with the overlapping annotation
  conflicting_label_name?: string | null;  // If skipped due to different class at same location
  status?: TrackingStatus;
  uncertain_reasons?: string[];
  signals?: TrackingSignals | null;
  next_expected_area_ratio?: number;
}

// Tracking uncertainty (issue #5). 'needs_review' = plausible but uncertain;
// a lost object is reported as an HTTP 400 error instead.
export type TrackingStatus = 'ok' | 'needs_review';

export interface TrackingSignals {
  embed_similarity: number;
  sam_score: number;
  bbox_area_ratio: number;
  mask_area_ratio: number | null;
  anchor_area_ratio: number | null;
  center_jump: number;
  expected_area_ratio: number;
  touches_border: boolean;
  dense_iou: number | null;
}

// Point for refinement
export interface RefinePoint {
  x: number;
  y: number;
  type: 'positive' | 'negative';
}

// Bounding box being drawn
export interface DrawingBbox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// ==================== Advanced ML Types ====================

// Instance found by find-all-instances
export interface FoundInstance {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  method: string;
  area_ratio?: number;
}

// Response from find-all-instances endpoint
export interface FindAllInstancesResult {
  instances: FoundInstance[];
  count: number;
}

// Advanced propagation result with method tracking
export interface PropagateAdvancedResult extends TrackIdentityFields {
  bbox: [number, number, number, number];
  mask_rle: object;
  polygon: number[];
  confidence: number;
  fallback_used: boolean;
  area_ratio: number;
  method: 'peak' | 'dense' | 'iou_match';
  iou_score: number | null;
  duplicate_skipped?: boolean;
  duplicate_iou?: number;
  conflicting_label_name?: string | null;  // If skipped due to different class at same location
  status?: TrackingStatus;
  uncertain_reasons?: string[];
  signals?: TrackingSignals | null;
  next_expected_area_ratio?: number;
}

// Propagation mode for advanced propagation
export type PropagationMode = 'peak' | 'dense' | 'auto';

// ==================== Backend tracking jobs (#3) ====================

export type TrackingJobStatus = 'pending' | 'running' | 'paused' | 'finished' | 'stopped' | 'error';
export type TrackingFailureMode = 'stop' | 'skip' | 'continue';

export interface TrackingJobRequest {
  start_index?: number;
  start_image_id?: number;
  end_index?: number | null;
  num_frames?: number;
  track_ids?: number[] | null;
  label_ids?: number[] | null;
  failure_mode?: TrackingFailureMode;
  performance?: boolean;
  mode?: PropagationMode;
  iou_verify?: boolean;
  iou_threshold?: number;
  use_cached_masks?: boolean;
  use_bbox_hint?: boolean;
  bbox_hint_scale?: number;
  prune_thin_artifacts?: boolean;
  size_min_ratio?: number;
  size_max_ratio?: number;
  top_k?: number;
  min_similarity?: number;
  duplicate_threshold?: number;
}

export interface TrackingJobPausedTrack {
  track_id: number;
  label_id: number;
  label_name: string;
}

export interface TrackingJob {
  id: string;
  project_id: number;
  status: TrackingJobStatus;
  start_index: number;
  end_index: number | null;
  current_index: number;
  current_image_id: number | null;
  total_frames: number;
  counts: Record<string, number>;
  tracks: Array<{ track_id: number; label_id: number; state: string; expected_area_ratio: number }>;
  paused: { image_id: number; image_index: number; tracks: TrackingJobPausedTrack[] } | null;
  settings: Record<string, unknown> & { performance?: boolean; failure_mode?: TrackingFailureMode };
  error: string | null;
  last_seq: number;
}

/** Event emitted by a tracking job (see ml/tracking_job.py). Extra fields depend on `type`. */
export interface TrackingJobEvent {
  seq: number;
  type:
    | 'started' | 'frame_done' | 'uncertain' | 'possible_id_switch' | 'object_lost'
    | 'paused_waiting_for_label' | 'reanchored' | 'object_resumed' | 'resumed'
    | 'finished' | 'error';
  job_id: string;
  ts: number;
  image_id?: number;
  image_index?: number;
  track_id?: number;
  label_name?: string;
  reason?: string;
  message?: string;
  reasons?: string[];
  tracks?: TrackingJobPausedTrack[];
  created?: Array<{ annotation_id: number; track_id: number; status: string }>;
  skipped?: Array<{ track_id: number }>;
  lost?: number[];
  progress?: { done: number; total: number };
  counts?: Record<string, number>;
  paused?: 'job' | 'object';
  [key: string]: unknown;
}
