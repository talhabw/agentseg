import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { PHASE_NAMES, useAgentStore } from './agentStore';
import { ActionLine, Controls, PhaseSteps, ProgressLine, ViewSwitch, Words } from './agentParts';
import { useAgentNow, useAgentVisible } from './agentView';

/**
 * Layout "Sidebar": Muse's card at the bottom of the sidebar, as before but
 * larger, with the stage it is in. Clicking it opens the run log.
 */
export function AgentPanel() {
  const visible = useAgentVisible();
  const view = useAgentStore((s) => s.view);
  const logOpen = useAgentStore((s) => s.logOpen);
  const setLogOpen = useAgentStore((s) => s.setLogOpen);
  const { run, action, phase, running, line, thinking, text, progress } = useAgentNow();
  if (!visible || !run) return null;

  if (view !== 'sidebar') {
    // Another layout shows the agent; the sidebar keeps only the switcher, so the layouts can be compared.
    return (
      <section className="flex shrink-0 items-center justify-between border-t px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Muse</span>
        <ViewSwitch />
      </section>
    );
  }

  return (
    <section className="flex shrink-0 flex-col border-t">
      <div className="px-4 pb-4 pt-3">
        <h3 className="mb-2.5 flex h-6 items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <span>Muse</span>
          <Separator className="flex-1" />
          <Controls />
        </h3>
        <button
          type="button"
          data-agent-anchor="agent-bubble"
          onClick={() => setLogOpen(!logOpen)}
          title="Open the run log"
          className="group relative block w-full overflow-hidden rounded-lg border bg-muted/20 px-4 py-3.5 text-left transition-colors hover:bg-muted/35"
          style={{ borderColor: running ? '#a594ff55' : undefined }}
        >
          <div className="mb-2 flex items-center gap-2">
            <PhaseSteps phase={phase} done={!running} compact />
            <span className="text-[11px] text-muted-foreground">{running ? (phase ? PHASE_NAMES[phase] : 'Starting') : 'Finished'}</span>
          </div>
          <div className="min-h-[60px]">
            {text ? (
              <Words key={line!.id} text={text}
                className={cn('line-clamp-4 text-[15px] leading-[22px]',
                  line!.kind === 'error' ? 'text-red-300/90' : thinking ? 'text-foreground/60' : 'text-foreground/95')}
              />
            ) : (
              <p className={cn('text-[15px] leading-[22px]', running ? 'agent-shimmer' : 'text-muted-foreground')}>
                {running ? 'Getting ready' : 'Finished'}
              </p>
            )}
          </div>
          {running && <ActionLine action={action} thinking={thinking} className="mt-2.5" />}
          <ProgressLine progress={progress} className="absolute inset-x-0 bottom-0" />
        </button>
        <div className="mt-2 flex justify-end"><ViewSwitch /></div>
      </div>
    </section>
  );
}
