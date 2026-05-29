import {
  Activity,
  Archive,
  Braces,
  CheckCircle,
  Cpu,
  Database,
  FileSearch,
  Gauge,
  GitBranch,
  Image as ImageIcon,
  Play,
  RefreshCw,
  Send,
  Server,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import { FormEvent, ReactNode, useEffect, useMemo, useState } from 'react';
import {
  askQuestion,
  getAdapters,
  getLatestProofSuite,
  getMemoryMap,
  getStatus,
  ingestPath,
  runHashHop,
  runProofSuite,
  uploadFiles,
} from './api';
import type {
  AdapterStatus,
  AnswerResponse,
  HashHopProof,
  IngestResponse,
  MemoryMap,
  ProofSuiteReport,
  RuntimeProfile,
  SystemStatus,
} from './types';

const profiles: Array<{ id: RuntimeProfile; label: string }> = [
  { id: 'live_demo', label: 'Live Demo' },
  { id: 'extreme_context', label: 'Extreme' },
  { id: 'library_100m', label: '100M Library' },
  { id: 'research_theater', label: 'Research' },
];

export function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [memoryMap, setMemoryMap] = useState<MemoryMap | null>(null);
  const [adapters, setAdapters] = useState<AdapterStatus[]>([]);
  const [profile, setProfile] = useState<RuntimeProfile>('library_100m');
  const [path, setPath] = useState('');
  const [cartridgeName, setCartridgeName] = useState('ely-eye-cartridge');
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [hashHop, setHashHop] = useState<HashHopProof | null>(null);
  const [proofSuite, setProofSuite] = useState<ProofSuiteReport | null>(null);
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setError(null);
    const [nextStatus, nextMap, nextAdapters, nextProofSuite] = await Promise.all([
      getStatus(),
      getMemoryMap(),
      getAdapters(),
      getLatestProofSuite().catch(() => null),
    ]);
    setStatus(nextStatus);
    setMemoryMap(nextMap);
    setAdapters(nextAdapters);
    setProofSuite(nextProofSuite);
  };

  useEffect(() => {
    let cancelled = false;
    async function loadInitial() {
      try {
        const [nextStatus, nextMap, nextAdapters, nextProofSuite] = await Promise.all([
          getStatus(),
          getMemoryMap(),
          getAdapters(),
          getLatestProofSuite().catch(() => null),
        ]);
        if (!cancelled) {
          setStatus(nextStatus);
          setMemoryMap(nextMap);
          setAdapters(nextAdapters);
          setProofSuite(nextProofSuite);
        }
      } catch (err) {
        if (!cancelled) {
          setError(errorMessage(err));
        }
      }
    }
    void loadInitial();
    return () => {
      cancelled = true;
    };
  }, []);

  const memoryLayers = useMemo(() => status?.memory.layers ?? [], [status]);
  const boundAdapters = useMemo(() => adapters.filter((adapter) => adapter.cartridge_bound).length, [adapters]);
  const passedProofChecks = useMemo(
    () => proofSuite?.checks.filter((check) => check.status === 'passed').length ?? 0,
    [proofSuite],
  );

  async function submitPath(event: FormEvent) {
    event.preventDefault();
    setBusy('ingest-path');
    setError(null);
    try {
      const result = await ingestPath(path, cartridgeName);
      setLastIngest(result);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitUpload(event: FormEvent) {
    event.preventDefault();
    setBusy('upload');
    setError(null);
    try {
      const result = await uploadFiles(files, cartridgeName);
      setLastIngest(result);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    setBusy('chat');
    setError(null);
    try {
      const result = await askQuestion(question, profile);
      setAnswer(result);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitHashHop(kind: 'text' | 'visual') {
    setBusy(`hashhop-${kind}`);
    setError(null);
    try {
      const result = await runHashHop(kind, kind === 'visual' ? 2 : 2, kind === 'visual' ? 262144 : 1010000);
      setHashHop(result);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitProofSuite() {
    setBusy('proof-suite');
    setError(null);
    try {
      const cartridgeId = memoryMap?.cartridges[0]?.cartridge_id;
      const result = await runProofSuite(cartridgeId);
      setProofSuite(result);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Ely-Eye v4.0</p>
          <h1>Local VLM Context OS</h1>
        </div>
        <button className="icon-button" title="Refresh status" onClick={() => refresh().catch((err) => setError(errorMessage(err)))}>
          <RefreshCw size={18} />
        </button>
      </header>

      {error && <div className="error-line">{error}</div>}

      <section className="kpi-grid" aria-label="system status">
        <Kpi icon={<Server size={18} />} label="Runtime" value={status?.runtime.available ? 'ready' : 'offline'} detail={formatRuntimeDetail(status)} />
        <Kpi icon={<Gauge size={18} />} label="GPU" value={status?.runtime.gpu_name ?? 'unknown'} detail={formatVram(status)} />
        <Kpi icon={<Database size={18} />} label="Library" value={formatNumber(status?.memory.library_token_equivalent ?? 0)} detail="token eq" />
        <Kpi icon={<Archive size={18} />} label="Cartridges" value={String(status?.memory.cartridges ?? 0)} detail={`${boundAdapters} adapters bound`} />
      </section>

      <section className="workspace-grid">
        <Panel title="Input Stream" icon={<Upload size={17} />}>
          <form className="stack" onSubmit={submitPath}>
            <label>
              Cartridge
              <input value={cartridgeName} onChange={(event) => setCartridgeName(event.target.value)} />
            </label>
            <label>
              Local path
              <input value={path} onChange={(event) => setPath(event.target.value)} />
            </label>
            <button className="action-button" disabled={!path || busy === 'ingest-path'}>
              <FileSearch size={16} />
              Ingest Path
            </button>
          </form>
          <form className="stack" onSubmit={submitUpload}>
            <label>
              Files
              <input type="file" multiple onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
            </label>
            <button className="action-button secondary" disabled={files.length === 0 || busy === 'upload'}>
              <Upload size={16} />
              Upload Files
            </button>
          </form>
          {lastIngest && (
            <div className="metric-strip">
              <span>{lastIngest.source_count} sources</span>
              <span>{lastIngest.atom_count} atoms</span>
              <span>{formatNumber(lastIngest.token_equivalent)} tokens</span>
            </div>
          )}
        </Panel>

        <Panel title="Context Compiler" icon={<Braces size={17} />}>
          <div className="segments">
            {profiles.map((item) => (
              <button
                key={item.id}
                className={profile === item.id ? 'segment active' : 'segment'}
                onClick={() => setProfile(item.id)}
                type="button"
              >
                {item.label}
              </button>
            ))}
          </div>
          <form className="stack" onSubmit={submitQuestion}>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={6} />
            <button className="action-button" disabled={!question || busy === 'chat'}>
              <Send size={16} />
              Ask
            </button>
          </form>
          {answer && (
            <div className="answer-box">
              <p>{answer.answer}</p>
              <div className="metric-strip">
                <span>{answer.context.hits.length} hits</span>
                <span>{Math.round(answer.verifier.citation_accuracy * 100)}% citations</span>
                <span>{answer.runtime_backend}</span>
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Memory OS" icon={<Database size={17} />}>
          <div className="layer-list">
            {memoryLayers.map((layer) => (
              <div className="layer-row" key={layer.layer}>
                <span>{layer.layer}</span>
                <strong>{layer.entries}</strong>
                <small>{layer.hit_count} hit / {layer.miss_count} miss</small>
              </div>
            ))}
            {memoryLayers.length === 0 && <div className="quiet-line">No cache events</div>}
          </div>
          <div className="atom-list">
            {(memoryMap?.atoms ?? []).slice(0, 8).map((atom) => (
              <div className="atom-row" key={atom.atom_id}>
                <span>{atom.modality}</span>
                <strong>{atom.atom_id}</strong>
                <small>{atom.source}</small>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Adapter Matrix" icon={<Cpu size={17} />}>
          <div className="adapter-list">
            {adapters.map((adapter) => (
              <div className="adapter-row" key={adapter.adapter_id}>
                <header>
                  <span>{adapter.kind.replaceAll('_', '-')}</span>
                  {adapter.cartridge_bound && <CheckCircle size={16} aria-label="bound" />}
                </header>
                <strong>{adapter.base_model}</strong>
                <div className="adapter-metrics">
                  <small>{adapter.training_method}</small>
                  <small>{adapter.precision}</small>
                  <small title={adapter.unsloth_model_type ?? adapter.framework}>{formatFramework(adapter.framework)}</small>
                  <small title={adapter.optimizer_family ?? adapter.optimizer ?? undefined}>{formatOptimizer(adapter)}</small>
                  <small title={adapter.gradient_checkpointing_mode ?? undefined}>{formatCheckpointing(adapter)}</small>
                  <small title={adapter.triton_version ?? undefined}>{adapter.triton_version ? `triton ${adapter.triton_version}` : 'triton pending'}</small>
                  <small>{formatBytes(adapter.weight_bytes)}</small>
                  <small>{formatNumber(adapter.trainable_params)} params</small>
                  <small>{adapter.final_loss?.toFixed(3) ?? 'loss pending'}</small>
                  <small>{adapter.training_trace_steps}/{adapter.max_steps} trace</small>
                  <small>{formatNumber(adapter.optimizer_update_count)} updates</small>
                  <small>{adapter.loss_delta == null ? 'loss delta pending' : `loss delta ${adapter.loss_delta.toFixed(3)}`}</small>
                  <small>{adapter.bf16 ? 'cuda bf16' : 'precision pending'}</small>
                  <small>{formatNumber(adapter.adapter_lora_tensor_count)} LoRA tensors</small>
                  <small>{formatNumber(adapter.adapter_nonzero_elements)} nonzero</small>
                  <small>{adapter.adapter_weights_finite ? 'finite weights' : 'weights pending'}</small>
                  <small>{adapter.adapter_weight_sha256 ? `sha ${shortHash(adapter.adapter_weight_sha256)}` : 'sha pending'}</small>
                  <small>{formatDuration(adapter.training_wall_seconds)}</small>
                </div>
              </div>
            ))}
            {adapters.length === 0 && <div className="quiet-line">No adapters found</div>}
          </div>
        </Panel>

        <Panel title="PRD Proof Suite" icon={<ShieldCheck size={17} />}>
          <div className="proof-suite-header">
            <button className="action-button secondary" onClick={submitProofSuite} disabled={busy === 'proof-suite'}>
              <Play size={16} />
              Run Suite
            </button>
            <div className={proofSuite?.status === 'passed' ? 'proof-pill passed' : 'proof-pill'}>
              {proofSuite ? `${passedProofChecks}/${proofSuite.checks.length} passed` : 'pending'}
            </div>
          </div>
          {proofSuite && (
            <div className="proof-list">
              {proofSuite.checks.map((check) => (
                <article className="proof-row" key={check.name}>
                  <header>
                    <strong>{check.name}</strong>
                    <span className={check.status === 'passed' ? 'proof-status passed' : 'proof-status'}>
                      {check.status}
                    </span>
                  </header>
                  <p>{check.detail}</p>
                  <small>{check.evidence[0]}</small>
                </article>
              ))}
              <div className="proof-box">
                <strong>{proofSuite.proof_id}</strong>
                <span>{proofSuite.status}</span>
                <small>{proofSuite.cartridge_id}</small>
                <small>{proofSuite.dna_after ?? proofSuite.dna_before}</small>
              </div>
            </div>
          )}
          {!proofSuite && <div className="quiet-line">Proof suite has not run yet</div>}
        </Panel>

        <Panel title="Evidence Viewer" icon={<ImageIcon size={17} />}>
          <div className="evidence-list">
            {(answer?.context.hits ?? []).slice(0, 8).map((hit) => (
              <article key={hit.atom.atom_id} className="evidence-row">
                <header>
                  <span>{hit.atom.modality}</span>
                  <strong>{hit.final_score.toFixed(3)}</strong>
                </header>
                <p>{hit.atom.text.slice(0, 280)}</p>
                <small>{hit.atom.source}</small>
              </article>
            ))}
            {!answer && <div className="quiet-line">Evidence appears after a compiled answer</div>}
          </div>
        </Panel>

        <Panel title="Visual HashHop Arena" icon={<GitBranch size={17} />}>
          <div className="button-row">
            <button className="action-button secondary" onClick={() => submitHashHop('visual')} disabled={busy === 'hashhop-visual'}>
              <Play size={16} />
              Visual 2-hop
            </button>
            <button className="action-button secondary" onClick={() => submitHashHop('text')} disabled={busy === 'hashhop-text'}>
              <Play size={16} />
              Text 2-hop
            </button>
          </div>
          {hashHop && (
            <div className="proof-box">
              <strong>{hashHop.proof_id}</strong>
              <span>{hashHop.kind}</span>
              <span>{hashHop.hops} hops</span>
              <span>{formatNumber(hashHop.token_equivalent)} token eq</span>
              <small>{hashHop.expected_target_id}</small>
            </div>
          )}
        </Panel>

        <Panel title="Cache Trace" icon={<Activity size={17} />}>
          <div className="trace-list">
            {memoryLayers.map((layer) => (
              <div key={layer.layer} className="trace-row">
                <span>{layer.layer}</span>
                <meter min={0} max={Math.max(1, layer.hit_count + layer.miss_count)} value={layer.hit_count} />
                <small>{formatBytes(layer.bytes_estimate)}</small>
              </div>
            ))}
            {memoryLayers.length === 0 && <div className="quiet-line">Waiting for retrieval</div>}
          </div>
        </Panel>
      </section>
    </main>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-header">
        <span className="panel-icon">{icon}</span>
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

function Kpi({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail: string }) {
  return (
    <section className="kpi">
      <span className="panel-icon">{icon}</span>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </section>
  );
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KiB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MiB`;
}

function formatDuration(value: number | null): string {
  if (typeof value !== 'number') {
    return 'time pending';
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  return `${Math.round(value / 60)}m`;
}

function shortHash(value: string): string {
  return value.slice(0, 8);
}

function formatOptimizer(adapter: AdapterStatus): string {
  if (adapter.optimizer === 'paged_adamw_8bit') {
    return 'paged 8-bit AdamW';
  }
  return adapter.optimizer_family ?? adapter.optimizer ?? 'optimizer pending';
}

function formatFramework(value: string): string {
  if (value === 'unsloth_patched_transformers_peft') {
    return 'Unsloth patch + PEFT';
  }
  return value;
}

function formatCheckpointing(adapter: AdapterStatus): string {
  if (adapter.gradient_checkpointing_mode === 'unsloth') {
    return 'ckpt unsloth';
  }
  return adapter.gradient_checkpointing ? 'ckpt on' : 'ckpt pending';
}

function formatRuntimeDetail(status: SystemStatus | null): string {
  const model = status?.runtime.model_id ?? 'Qwen/Qwen3.5-9B';
  const adapter = status?.runtime.adapter_kind?.replaceAll('_', '-');
  return adapter ? `${model} · ${adapter}` : model;
}

function formatVram(status: SystemStatus | null): string {
  const total = status?.runtime.vram_total_mb;
  const used = status?.runtime.vram_used_mb;
  if (typeof total === 'number' && typeof used === 'number') {
    return `${used} / ${total} MiB`;
  }
  return status?.runtime.cuda ?? 'CUDA';
}
