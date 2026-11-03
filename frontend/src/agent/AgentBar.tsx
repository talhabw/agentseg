import { cn } from '@/lib/utils';
import { AGENT_COLOR, modelDisplayName, useAgentStore } from './agentStore';
import { ActionLine, Controls, PhaseSteps, ProgressLine, ViewSwitch, Words } from './agentParts';
import { useAgentNow, useAgentVisible } from './agentView';

/**
 * Layout "Top bar": a strip under the header, the whole width of the window.
 * Who is working, what it says now, the tool call, the stage, and the controls.
 * The thin line along the bottom fills while SAM tracks.
 */
export function AgentBar() {
  const visible = useAgentVisible();
  const view = useAgentStore((s) => s.view);
  const { run, action, phase, running, line, thinking, text, progress } = useAgentNow();
  if (!visible || view !== 'bar' || !run) return null;
  return (
    <div className="relative z-20 shrink-0 border-b bg-[#a594ff0d]" data-agent-anchor="agent-bubble">
      <div className="flex min-h-[56px] items-center gap-4 px-4 py-2">
        <div className="flex w-[132px] shrink-0 items-center gap-2">
          <span className={cn('h-2.5 w-2.5 shrink-0 rounded-full', running && 'agent-pulse')} style={{ background: running ? AGENT_COLOR : 'hsl(240 5% 45%)' }} />
          <div className="min-w-0 leading-tight">
            <div className="truncate text-[13px] font-semibold" style={{ color: AGENT_COLOR }}>{modelDisplayName(run.model)}</div>
            <div className="text-[10.5px] text-muted-foreground">{running ? 'labeling' : run.status === 'done' ? 'finished' : run.status}</div>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          {text ? (
            <Words key={line!.id} text={text}
              className={cn('line-clamp-2 text-[14.5px] leading-5',
                line!.kind === 'error' ? 'text-red-300/90' : thinking ? 'text-foreground/60' : 'text-foreground')}
            />
          ) : (
            <p className={cn('text-[14.5px] leading-5', running ? 'agent-shimmer' : 'text-muted-foreground')}>{running ? 'Getting ready' : 'Finished'}</p>
          )}
          {running && <ActionLine action={action} thinking={thinking} className="mt-1 max-w-[560px]" />}
        </div>
        <div className="hidden shrink-0 xl:block"><PhaseSteps phase={phase} done={!running} /></div>
        <div className="shrink-0 xl:hidden"><PhaseSteps phase={phase} done={!running} compact /></div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Controls />
          <ViewSwitch />
        </div>
      </div>
      <ProgressLine progress={progress} className="absolute inset-x-0 bottom-0" />
    </div>
  );
}
