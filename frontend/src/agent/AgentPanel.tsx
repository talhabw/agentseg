import { useLayoutEffect, useRef } from 'react';
import { ChevronDown } from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useUIStore } from '../stores/uiStore';
import { AGENT_COLOR, useAgentStore, type AgentAction, type FeedItem } from './agentStore';
import { setReplaySpeed, toggleReplay } from './replay';

/**
 * Muse's corner of the sidebar. By default one bubble: what it is doing now,
 * in its own words, and the tool call under it. Clicking the bubble opens the
 * whole story below it (sentences and the steps that changed the project).
 * No icons, no thumbnails (the canvas already shows what it sees).
 */

function Words({ text, className }: { text: string; className?: string }) {
  // Words settle in one after another, like the sentence is being said.
  return (
    <p className={className}>
      {text.split(/(\s+)/).map((part, i) => (/^\s+$/.test(part)
        ? part
        : <span key={i} className="agent-word" style={{ animationDelay: `${Math.min(i, 60) * 14}ms` }}>{part}</span>))}
    </p>
  );
}

function ActionLine({ action, thinking }: { action: AgentAction | null; thinking: boolean }) {
  if (!action) return null;
  const running = action.state === 'running';
  return (
    <div key={action.id} className="agent-rise mt-2.5 flex min-w-0 items-baseline gap-2 font-mono text-[11px] leading-4">
      <span className="shrink-0" style={{ color: AGENT_COLOR, opacity: thinking && !running ? 0.6 : 1 }}>{action.tool}</span>
      {action.detail && (
        <span className={cn('min-w-0 truncate', running ? 'agent-shimmer' : 'text-muted-foreground')}>{action.detail}</span>
      )}
      {action.result && (
        <span key={action.result} className={cn('ml-auto shrink-0 tabular-nums',
          action.state === 'failed' ? 'text-red-300/80' : 'text-foreground/70')}
        >
          {action.result}
        </span>
      )}
    </div>
  );
}

function Report({ text }: { text: string }) {
  const lines = text.split('\n').map((l) => l.replace(/^\s*[-*]\s+/, '– ')).filter((l, i, all) => l.trim() || (i > 0 && all[i - 1].trim()));
  return (
    <div className="mt-3 rounded-md border bg-muted/30 px-3 py-2.5 text-[12.5px] leading-5 text-foreground/90 whitespace-pre-line">
      {lines.join('\n')}
    </div>
  );
}

function Row({ item }: { item: FeedItem }) {
  if (item.kind === 'thought') {
    return <p className="mt-3 text-[13px] leading-5 text-foreground/90 first:mt-0">{item.text}</p>;
  }
  if (item.kind === 'report') return <Report text={item.text} />;
  if (item.kind === 'error') {
    return <p className="mt-1.5 text-xs leading-5 text-red-300/80">{item.text}</p>;
  }
  return (
    <div className="mt-1.5 flex items-baseline gap-3 text-xs leading-5">
      <span className={cn('min-w-0 flex-1 truncate',
        item.state === 'running' ? 'agent-shimmer' : item.state === 'failed' ? 'text-muted-foreground/60 line-through' : 'text-muted-foreground')}
      >
        {item.text}
      </span>
      {item.meta && <span className="shrink-0 tabular-nums text-muted-foreground/70">{item.meta}</span>}
    </div>
  );
}

function Log({ feed }: { feed: FeedItem[] }) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [feed]);
  return (
    <div
      ref={scroller}
      className="min-h-0 flex-1 overflow-y-auto px-4 pb-4 pt-1 [mask-image:linear-gradient(to_bottom,transparent_0,black_20px)]"
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      <div className="pt-3">
        {feed.map((item) => <Row key={item.id} item={item} />)}
      </div>
    </div>
  );
}

function Controls() {
  const run = useAgentStore((s) => s.run);
  const replay = useAgentStore((s) => s.replay);
  const { followAgent, setFollowAgent } = useUIStore();
  if (!run) return null;
  const running = run.status === 'running';
  const button = 'rounded px-1.5 py-0.5 hover:text-foreground';
  return (
    <div className="flex shrink-0 items-center gap-0.5 text-[11px] font-normal normal-case tracking-normal text-muted-foreground">
      {replay && (
        <>
          <span className="mr-1" title={replay.synthetic ? 'A hand-made demo recording built from a real run, not a real agent run' : 'A recorded run'}>
            {replay.synthetic ? 'Synthetic replay' : 'Replay'}
          </span>
          {[1, 2, 4].map((speed) => (
            <button key={speed} type="button" onClick={() => setReplaySpeed(speed)}
              className={cn(button, 'tabular-nums', replay.speed === speed && 'bg-muted text-foreground')}
            >
              {speed}×
            </button>
          ))}
          <button type="button" onClick={toggleReplay} className={button}>
            {replay.finished ? 'Again' : replay.playing ? 'Pause' : 'Play'}
          </button>
        </>
      )}
      {running && !followAgent && (
        <button type="button" onClick={() => setFollowAgent(true)} className={cn(button, 'bg-muted text-foreground')}
          title="Go back to the frame Muse works on"
        >
          Follow
        </button>
      )}
      {running && !replay && run.id && (
        <button type="button" className={button}
          onClick={() => { void fetch(`/api/agent/runs/${run.id}/stop`, { method: 'POST' }); }}
        >
          Stop
        </button>
      )}
      {!running && !replay && (
        <button type="button" onClick={() => useAgentStore.getState().dismiss()} className={button}>Close</button>
      )}
    </div>
  );
}

/** The sentence in the bubble: the newest thing the agent said (the report's first paragraph at the end). */
function currentLine(feed: FeedItem[]): FeedItem | undefined {
  for (let i = feed.length - 1; i >= 0; i--) {
    const item = feed[i];
    if (item.kind === 'thought' || item.kind === 'report' || item.kind === 'error') return item;
  }
  return undefined;
}

export function AgentPanel() {
  const run = useAgentStore((s) => s.run);
  const feed = useAgentStore((s) => s.feed);
  const action = useAgentStore((s) => s.action);
  const activity = useAgentStore((s) => s.activity);
  const dismissed = useAgentStore((s) => s.dismissed);
  const logOpen = useAgentStore((s) => s.logOpen);
  const setLogOpen = useAgentStore((s) => s.setLogOpen);

  if (!run || (dismissed && run.status !== 'running')) return null;
  const running = run.status === 'running';
  const line = currentLine(feed);
  // Between calls the model is thinking: its last sentence has been acted on.
  const thinking = running && activity.kind === 'thinking' && action?.state !== 'running';
  const text = line?.kind === 'report' ? line.text.split(/\n\s*\n/)[0] : line?.text;

  return (
    <section className={cn('flex flex-col border-t', logOpen ? 'min-h-0 flex-1' : 'shrink-0')}>
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
          aria-expanded={logOpen}
          title={logOpen ? 'Hide the steps' : 'Show every step'}
          className="group relative block w-full rounded-lg border bg-muted/20 px-3.5 py-3 text-left transition-colors hover:bg-muted/35"
        >
          <ChevronDown
            className={cn('absolute right-2.5 top-3 h-3.5 w-3.5 text-muted-foreground/50 transition-transform group-hover:text-muted-foreground',
              logOpen && 'rotate-180')}
          />
          <div className="min-h-[40px] pr-5">
            {text ? (
              <Words
                key={line!.id}
                text={text}
                className={cn('text-[13.5px] leading-5',
                  line!.kind === 'error' ? 'text-red-300/90' : thinking ? 'agent-shimmer' : 'text-foreground/95',
                  line!.kind === 'report' && 'line-clamp-3')}
              />
            ) : (
              <p className={cn('text-[13.5px] leading-5', running ? 'agent-shimmer' : 'text-muted-foreground')}>
                {running ? 'Getting ready' : 'Finished'}
              </p>
            )}
          </div>
          {running && <ActionLine action={action} thinking={thinking} />}
        </button>
      </div>
      {logOpen && <Log feed={feed} />}
    </section>
  );
}
