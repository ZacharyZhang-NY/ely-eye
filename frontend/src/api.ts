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

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail));
  }
  return (await response.json()) as T;
}

export function getStatus(): Promise<SystemStatus> {
  return request<SystemStatus>('/api/status');
}

export function getMemoryMap(): Promise<MemoryMap> {
  return request<MemoryMap>('/api/memory-map');
}

export function getAdapters(): Promise<AdapterStatus[]> {
  return request<AdapterStatus[]>('/api/adapters');
}

export function ingestPath(path: string, cartridgeName: string): Promise<IngestResponse> {
  return request<IngestResponse>('/api/ingest/path', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, cartridge_name: cartridgeName || null }),
  });
}

export function uploadFiles(files: File[], cartridgeName: string): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append('files', file);
  }
  const query = new URLSearchParams();
  if (cartridgeName) {
    query.set('cartridge_name', cartridgeName);
  }
  return request<IngestResponse>(`/api/ingest/upload?${query.toString()}`, {
    method: 'POST',
    body: form,
  });
}

export function askQuestion(question: string, profile: RuntimeProfile): Promise<AnswerResponse> {
  return request<AnswerResponse>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, profile }),
  });
}

export function runHashHop(kind: 'text' | 'visual', hops: number, tokenEquivalent: number): Promise<HashHopProof> {
  return request<HashHopProof>('/api/hashhop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, hops, token_equivalent: tokenEquivalent }),
  });
}

export function runProofSuite(cartridgeId?: string): Promise<ProofSuiteReport> {
  return request<ProofSuiteReport>('/api/proof-suite', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cartridge_id: cartridgeId ?? null }),
  });
}

export function getLatestProofSuite(): Promise<ProofSuiteReport> {
  return request<ProofSuiteReport>('/api/proof-suite/latest');
}
