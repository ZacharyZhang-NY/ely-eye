export type RuntimeProfile = 'live_demo' | 'extreme_context' | 'library_100m' | 'research_theater';

export interface RuntimeStatus {
  backend: string;
  model_id: string;
  available: boolean;
  detail: string;
  adapter_path: string | null;
  adapter_kind: string | null;
  adapter_id: string | null;
  adapter_sha256: string | null;
  gpu_name: string | null;
  cuda: string | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
}

export interface CacheLayerStatus {
  layer: string;
  entries: number;
  bytes_estimate: number;
  hit_count: number;
  miss_count: number;
  last_event_at: string | null;
}

export interface MemoryStatus {
  active_atoms: number;
  active_token_equivalent: number;
  cartridges: number;
  library_token_equivalent: number;
  layers: CacheLayerStatus[];
}

export interface SystemStatus {
  app: string;
  version: string;
  runtime: RuntimeStatus;
  memory: MemoryStatus;
  profiles: Record<string, Record<string, unknown>>;
  data_home: string;
}

export interface AdapterStatus {
  name: string;
  kind: string;
  proof_version: number;
  training_method: string;
  precision: string;
  framework: string;
  base_model: string;
  path: string;
  adapter_id: string;
  sample_count: number;
  max_steps: number;
  trainable_params: number;
  final_loss: number | null;
  sha256: string;
  weight_bytes: number;
  safetensor_keys: number;
  cartridge_id: string | null;
  cartridge_bound: boolean;
  training_trace_path: string | null;
  training_trace_sha256: string | null;
  training_trace_steps: number;
  training_proof_path: string | null;
  training_proof_sha256: string | null;
  optimizer: string | null;
  optimizer_family: string | null;
  gradient_checkpointing: boolean;
  gradient_checkpointing_mode: string | null;
  autocast_dtype: string | null;
  unsloth_version: string | null;
  triton_version: string | null;
  xformers_version: string | null;
  unsloth_model_type: string | null;
  optimizer_update_count: number;
  max_optimizer_update_l2: number | null;
  loss_delta: number | null;
  training_summary_path: string | null;
  training_wall_seconds: number | null;
  training_device: string | null;
  bf16: boolean;
  torch_version: string | null;
  cuda_version: string | null;
  adapter_weight_sha256: string | null;
  adapter_tensor_count: number;
  adapter_lora_tensor_count: number;
  adapter_total_elements: number;
  adapter_nonzero_elements: number;
  adapter_max_abs: number | null;
  adapter_weights_finite: boolean;
}

export interface IngestResponse {
  source_count: number;
  atom_count: number;
  cartridge_id: string | null;
  token_equivalent: number;
}

export interface LayoutBox {
  x: number;
  y: number;
  w: number;
  h: number;
  page?: number | null;
  frame_second?: number | null;
}

export interface EvidenceAtom {
  atom_id: string;
  modality: string;
  source: string;
  text: string;
  image_ref: string | null;
  layout: LayoutBox | null;
  token_equivalent: number;
}

export interface RetrievalHit {
  atom: EvidenceAtom;
  sparse_score: number;
  dense_score: number;
  graph_score: number;
  final_score: number;
}

export interface AnswerResponse {
  answer: string;
  citations: string[];
  context: {
    hits: RetrievalHit[];
    token_equivalent: number;
    cache_trace_id: string;
  };
  verifier: {
    cited_atom_ids: string[];
    missing_atom_ids: string[];
    citation_accuracy: number;
    contradiction_notes: string[];
    confidence: number;
  };
  runtime_backend: string;
}

export interface HashHopProof {
  proof_id: string;
  kind: string;
  hops: number;
  token_equivalent: number;
  query_id: string;
  expected_target_id: string;
  model_target_id: string | null;
  passed: boolean | null;
  artifacts: string[];
  created_at: string;
}

export interface ProofCheck {
  name: string;
  requirement: string;
  status: 'passed' | 'failed';
  evidence: string[];
  detail: string;
}

export interface ProofSuiteReport {
  proof_id: string;
  cartridge_id: string;
  status: 'passed' | 'failed';
  created_at: string;
  checks: ProofCheck[];
  artifacts: Record<string, string>;
  dna_before: string | null;
  dna_after: string | null;
}

export interface MemoryMap {
  atoms: EvidenceAtom[];
  layers: CacheLayerStatus[];
  cartridges: Array<{
    cartridge_id: string;
    name: string;
    token_equivalent: number;
    dna: string | null;
  }>;
}
