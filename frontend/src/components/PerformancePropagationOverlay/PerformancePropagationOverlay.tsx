import { useState, useEffect } from 'react';
import { useUIStore } from '../../stores/uiStore';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { resumeTracking, stopTracking } from '../../utils/trackingRunner';

export function PerformancePropagationOverlay() {
  const performanceProgress = useUIStore((s) => s.performanceProgress);
  const [now, setNow] = useState(Date.now);
  const isActive = !!performanceProgress;

  // Update clock every second for elapsed/ETA display
  // Only depend on isActive, not the whole object, to avoid resetting the interval on every progress update
  useEffect(() => {
    if (!isActive) return;
    setNow(Date.now()); // Reset clock when propagation starts
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isActive]);

  if (!performanceProgress) return null;

  const {
    current,
    total,
    successCount,
    failedCount,
    skippedCount,
    startTime,
    failedImageIndex,
    failedLabels,
    performance,
  } = performanceProgress;

  const elapsed = (now - startTime) / 1000;
  const rate = current > 0 ? elapsed / current : 0;
  const remaining = rate > 0 ? Math.ceil((total - current) * rate) : 0;
  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  const isStopped = failedImageIndex !== null;

  const handleStop = () => {
    void stopTracking();
    useUIStore.getState().setAutoNext(false);
  };

  const handleResume = () => {
    void resumeTracking();
  };

  // Full-screen only while a performance-mode job runs; otherwise (normal mode,
  // or paused waiting for a new label) a compact card so the canvas stays usable.
  const fullScreen = performance && !isStopped;

  return (
    <div className={fullScreen
      ? 'fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center'
      : 'fixed bottom-4 right-4 z-[100]'}>
      <Card className={fullScreen ? 'w-[480px] shadow-2xl' : 'w-[360px] shadow-2xl'}>
        <CardContent className="pt-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {isStopped ? 'Tracking Paused' : performance ? 'Performance Tracking' : 'Tracking'}
            </h2>
            <span className="text-sm text-muted-foreground">
              {percent}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-muted rounded-full h-3 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                isStopped ? 'bg-yellow-500' : 'bg-primary'
              }`}
              style={{ width: `${percent}%` }}
            />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="text-center">
              <div className="font-mono text-lg">{current}/{total}</div>
              <div className="text-muted-foreground">Images</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-lg text-green-500">{successCount}</div>
              <div className="text-muted-foreground">Success</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-lg text-red-500">{failedCount}</div>
              <div className="text-muted-foreground">Failed</div>
            </div>
          </div>

          {/* Time stats */}
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Elapsed: {formatTime(elapsed)}</span>
            <span>Rate: {rate > 0 ? `${rate.toFixed(1)}s/img` : '...'}</span>
            <span>ETA: {remaining > 0 ? formatTime(remaining) : '...'}</span>
          </div>

          {/* Failure info */}
          {isStopped && failedLabels.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-md p-3 text-sm">
              <div className="font-medium text-yellow-600 dark:text-yellow-400 mb-1">
                Failed to track at image {(failedImageIndex ?? 0) + 1}:
              </div>
              <div className="text-muted-foreground">
                {failedLabels.join(', ')}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Draw a new box for each lost object on this image (or re-anchor the track), then resume.
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            {isStopped ? (
              <>
                <Button variant="outline" className="flex-1" onClick={handleStop}>
                  Stop
                </Button>
                <Button className="flex-1" onClick={handleResume}>
                  Resume
                </Button>
              </>
            ) : (
              <Button variant="destructive" className="w-full" onClick={handleStop}>
                Stop Tracking
              </Button>
            )}
          </div>

          {skippedCount > 0 && (
            <div className="text-xs text-muted-foreground text-center">
              {skippedCount} images skipped (duplicates)
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}
