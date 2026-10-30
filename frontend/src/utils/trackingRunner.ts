// Thin client for backend tracking jobs (issue #3). The frame loop, duplicate
// handling, fallback and persistence all run in the backend; this module only
// starts/stops/resumes jobs and mirrors their events into the UI stores.
import * as api from '../api/client';
import type { TrackingJob, TrackingJobEvent } from '../types';
import { useAnnotationStore } from '../stores/annotationStore';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';

let currentJobId: string | null = null;
let starting = false;
let unsubscribe: (() => void) | null = null;

export function getTrackingJobId(): string | null {
  return currentJobId;
}

/** A job is running (not paused): the UI should not accept edits. */
export function isTrackingJobRunning(): boolean {
  const progress = useUIStore.getState().performanceProgress;
  return currentJobId !== null && (progress?.failedImageIndex ?? null) === null;
}

export function isTrackingJobPaused(): boolean {
  const progress = useUIStore.getState().performanceProgress;
  return currentJobId !== null && progress?.failedImageIndex != null;
}

function notify(title: string, body: string) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  if (Notification.permission !== 'granted') return;
  try {
    new Notification(title, { body, tag: 'autoseg-tracking', requireInteraction: true });
  } catch {
    // best-effort
  }
}

/** Start tracking from the current image: one frame, or to the end (auto). */
export async function startTracking({ toEnd }: { toEnd: boolean }): Promise<void> {
  if (currentJobId || starting) return;
  const { images, currentImageIndex, project } = useProjectStore.getState();
  const ui = useUIStore.getState();
  if (!project) return;
  if (currentImageIndex >= images.length - 1) {
    ui.addToast('Already at last image', 'info');
    return;
  }
  if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission().catch(() => {});
  }

  ui.setIsPropagating(true);
  starting = true;
  try {
    if (!ui.propagationLoaded) {
      ui.setStatusMessage('Loading tracking models...');
      await ui.loadPropagation();
    }
    const s = useUIStore.getState();
    const job = await api.startTrackingJob({
      start_index: currentImageIndex,
      ...(toEnd ? {} : { num_frames: 1 }),
      failure_mode: s.propagationFailureMode,
      performance: s.performanceMode && toEnd,
      mode: s.propagationMode,
      iou_verify: s.iouVerify,
      iou_threshold: s.iouThreshold,
      use_bbox_hint: s.useBBoxHint,
      bbox_hint_scale: s.bboxHintScale,
      prune_thin_artifacts: s.pruneThinArtifacts,
      size_min_ratio: s.sizeMinRatio,
      size_max_ratio: s.sizeMaxRatio,
      top_k: s.topK,
      duplicate_threshold: s.trackingDuplicateThreshold,
    });
    attach(job, 0);
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status;
    if (status === 409) {
      await reattachActiveTrackingJob();
      useUIStore.getState().addToast('A tracking job is already running - attached to it', 'info');
      return;
    }
    console.error('Failed to start tracking job:', err);
    useUIStore.getState().setIsPropagating(false);
    useUIStore.getState().addToast('Failed to start tracking', 'error');
  } finally {
    starting = false;
  }
}

export async function stopTracking(): Promise<void> {
  const jobId = currentJobId;
  if (!jobId) return;
  useUIStore.getState().setStatusMessage('Stopping tracking...');
  try {
    await api.stopTrackingJob(jobId);
  } catch (err) {
    console.error('Failed to stop tracking job:', err);
    cleanup();
  }
}

export async function resumeTracking(): Promise<void> {
  const jobId = currentJobId;
  if (!jobId) return;
  try {
    await api.resumeTrackingJob(jobId);
    useUIStore.getState().updatePerformanceProgress({ failedImageIndex: null, failedLabels: [] });
  } catch (err) {
    console.error('Failed to resume tracking job:', err);
    useUIStore.getState().addToast('Failed to resume tracking', 'error');
  }
}

/** Re-attach to a job that is still running in the backend (e.g. after a reload). */
export async function reattachActiveTrackingJob(): Promise<void> {
  if (currentJobId) return;
  try {
    const job = await api.getActiveTrackingJob();
    if (job) attach(job, job.last_seq);
  } catch {
    // no project / backend down: nothing to attach to
  }
}

function attach(job: TrackingJob, after: number) {
  currentJobId = job.id;
  const ui = useUIStore.getState();
  ui.setIsPropagating(true);
  ui.setPerformanceProgress({
    current: job.counts.frames_done ?? 0,
    total: job.total_frames,
    successCount: 0,
    failedCount: 0,
    skippedCount: 0,
    startTime: Date.now(),
    failedImageIndex: job.paused?.image_index ?? null,
    failedLabels: job.paused?.tracks.map((t) => t.label_name) ?? [],
    performance: !!job.settings.performance,
  });
  unsubscribe?.();
  unsubscribe = api.subscribeTrackingJob(job.id, handleEvent, after);
}

function cleanup() {
  unsubscribe?.();
  unsubscribe = null;
  currentJobId = null;
  useUIStore.getState().setIsPropagating(false);
  useUIStore.getState().setPerformanceProgress(null);
}

/** Drop cached annotations for an image; reload if it is on screen. */
function refreshImage(imageId: number | undefined) {
  if (imageId == null) return;
  const store = useAnnotationStore.getState();
  store.invalidateAnnotations(imageId);
  const { images, currentImageIndex } = useProjectStore.getState();
  if (images[currentImageIndex]?.id === imageId) {
    void store.loadAnnotations(imageId).catch(() => {});
  }
}

function handleEvent(ev: TrackingJobEvent) {
  const ui = useUIStore.getState();
  const progress = ui.performanceProgress;
  const annotations = useAnnotationStore.getState();

  switch (ev.type) {
    case 'started':
      refreshImage(ev.image_id); // untracked start-frame labels got tracks
      break;
    case 'frame_done': {
      annotations.invalidateAnnotations(ev.image_id ?? -1);
      ui.updatePerformanceProgress({
        current: ev.progress?.done ?? progress?.current ?? 0,
        total: ev.progress?.total ?? progress?.total ?? 0,
        successCount: (progress?.successCount ?? 0) + ((ev.created?.length ?? 0) > 0 ? 1 : 0),
        failedCount: (progress?.failedCount ?? 0) + ((ev.lost?.length ?? 0) > 0 ? 1 : 0),
        skippedCount: (progress?.skippedCount ?? 0) + (ev.skipped?.length ?? 0),
      });
      if (!progress?.performance && ev.image_index != null) {
        useProjectStore.getState().setCurrentImageIndex(ev.image_index);
      }
      ui.setStatusMessage(
        `Tracked frame ${(ev.image_index ?? 0) + 1}: ${ev.created?.length ?? 0} created` +
          ((ev.skipped?.length ?? 0) > 0 ? `, ${ev.skipped?.length} duplicate(s) skipped` : ''),
      );
      break;
    }
    case 'object_lost':
      if (ev.paused === 'object') {
        ui.addToast(`Lost ${ev.label_name ?? 'object'} at frame ${(ev.image_index ?? 0) + 1} (${ev.reason})`, 'warning', 3000);
      }
      break;
    case 'possible_id_switch':
      ui.addToast(`Possible ID switch at frame ${(ev.image_index ?? 0) + 1} (flagged for review)`, 'warning', 2500);
      break;
    case 'paused_waiting_for_label': {
      const names = (ev.tracks ?? []).map((t) => t.label_name);
      if (ev.image_index != null) useProjectStore.getState().setCurrentImageIndex(ev.image_index);
      refreshImage(ev.image_id);
      ui.updatePerformanceProgress({ failedImageIndex: ev.image_index ?? 0, failedLabels: names });
      const first = ev.tracks?.[0];
      if (first) {
        annotations.selectLabel(first.label_id);
        ui.setMode('draw');
      }
      ui.setStatusMessage(`Tracking paused - lost: ${names.join(', ')}`);
      notify('Tracking paused', `Could not track: ${names.join(', ')}`);
      void import('sonner').then(({ toast }) =>
        toast.error('Tracking paused', {
          description: `Could not track: ${names.slice(0, 3).join(', ')}${names.length > 3 ? ` and ${names.length - 3} more` : ''}. Draw a new box for each lost object, then resume.`,
          duration: 10000,
        }),
      );
      break;
    }
    case 'resumed':
      ui.updatePerformanceProgress({ failedImageIndex: null, failedLabels: [] });
      break;
    case 'reanchored':
      refreshImage(ev.image_id);
      break;
    case 'finished': {
      const counts = ev.counts ?? {};
      if (progress?.performance && typeof ev.current_index === 'number') {
        useProjectStore.getState().setCurrentImageIndex(ev.current_index);
      }
      const msg = `Tracking ${ev.reason === 'stopped' ? 'stopped' : 'complete'}: ${counts.created ?? 0} created, ${counts.lost ?? 0} lost, ${counts.uncertain ?? 0} need review`;
      ui.setStatusMessage(msg);
      if (ev.reason !== 'stopped' || (progress?.total ?? 0) > 1) {
        ui.addToast(msg, (counts.lost ?? 0) > 0 ? 'warning' : 'success', 3000);
      }
      cleanup();
      break;
    }
    case 'error':
      ui.addToast(`Tracking failed: ${ev.message}`, 'error', 6000);
      cleanup();
      break;
  }
}
