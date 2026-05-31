package domain

type CacheLayer struct {
	Layer         string `json:"layer"`
	Entries       int    `json:"entries"`
	BytesEstimate int64  `json:"bytes_estimate"`
	HitCount      int    `json:"hit_count"`
	MissCount     int    `json:"miss_count"`
	LastEventAt   string `json:"last_event_at,omitempty"`
}

type Status struct {
	App                    string       `json:"app"`
	ProjectRoot            string       `json:"project_root"`
	DataHome               string       `json:"data_home"`
	Database               string       `json:"database"`
	AtomCount              int          `json:"atom_count"`
	SourceCount            int          `json:"source_count"`
	CartridgeCount         int          `json:"cartridge_count"`
	ActiveTokenEquivalent  int64        `json:"active_token_equivalent"`
	LibraryTokenEquivalent int64        `json:"library_token_equivalent"`
	CacheLayers            []CacheLayer `json:"cache_layers"`
}

type Cartridge struct {
	CartridgeID     string `json:"cartridge_id"`
	Name            string `json:"name"`
	RootPath        string `json:"root_path"`
	DNA             string `json:"dna,omitempty"`
	TokenEquivalent int64  `json:"token_equivalent"`
	CreatedAt       string `json:"created_at"`
	Manifest        any    `json:"manifest,omitempty"`
}

type CartridgeDetail struct {
	Cartridge
	Artifacts          map[string]string `json:"artifacts"`
	MemoryCapsuleIndex any               `json:"memory_capsule_index,omitempty"`
	AssetReport        any               `json:"asset_report,omitempty"`
}

type Atom struct {
	AtomID          string `json:"atom_id"`
	SourceID        string `json:"source_id"`
	Modality        string `json:"modality"`
	Source          string `json:"source"`
	Time            string `json:"time"`
	Text            string `json:"text"`
	ImageRef        string `json:"image_ref,omitempty"`
	Layout          any    `json:"layout,omitempty"`
	Relations       any    `json:"relations,omitempty"`
	Trust           any    `json:"trust,omitempty"`
	TokenEquivalent int64  `json:"token_equivalent"`
	Metadata        any    `json:"metadata,omitempty"`
}

type AtomHit struct {
	Atom        Atom    `json:"atom"`
	SparseScore float64 `json:"sparse_score"`
	FinalScore  float64 `json:"final_score"`
}

type AtomSummary struct {
	AtomID          string `json:"atom_id"`
	SourceID        string `json:"source_id"`
	Modality        string `json:"modality"`
	Source          string `json:"source"`
	Time            string `json:"time"`
	TextPreview     string `json:"text_preview"`
	ImageRef        string `json:"image_ref,omitempty"`
	TokenEquivalent int64  `json:"token_equivalent"`
}

type AtomSearchHit struct {
	Atom        AtomSummary `json:"atom"`
	SparseScore float64     `json:"sparse_score"`
	FinalScore  float64     `json:"final_score"`
}

type EvidencePack struct {
	Question         string          `json:"question"`
	Profile          string          `json:"profile"`
	TokenBudget      int64           `json:"token_budget"`
	TokenEquivalent  int64           `json:"token_equivalent"`
	CacheTraceID     string          `json:"cache_trace_id"`
	Hits             []AtomSearchHit `json:"hits"`
	PackedText       string          `json:"packed_text"`
	VerifierContract []string        `json:"verifier_contract"`
}

type ProofSuite struct {
	ProofID     string `json:"proof_id"`
	CartridgeID string `json:"cartridge_id"`
	Status      string `json:"status"`
	CreatedAt   string `json:"created_at"`
	Path        string `json:"path"`
	Payload     any    `json:"payload,omitempty"`
}

type HashHopProof struct {
	ProofID          string   `json:"proof_id"`
	Kind             string   `json:"kind"`
	Hops             int      `json:"hops"`
	TokenEquivalent  int64    `json:"token_equivalent"`
	QueryID          string   `json:"query_id"`
	ExpectedTargetID string   `json:"expected_target_id"`
	ModelTargetID    string   `json:"model_target_id,omitempty"`
	Passed           *bool    `json:"passed"`
	Artifacts        []string `json:"artifacts,omitempty"`
	CreatedAt        string   `json:"created_at"`
	Path             string   `json:"path"`
}

type HashHopBucket struct {
	Kind            string `json:"kind"`
	Hops            int    `json:"hops"`
	TokenEquivalent int64  `json:"token_equivalent"`
	Total           int    `json:"total"`
	Passed          int    `json:"passed"`
}

type HashHopReport struct {
	Total   int             `json:"total"`
	Passed  int             `json:"passed"`
	Buckets []HashHopBucket `json:"buckets"`
	Proofs  []HashHopProof  `json:"proofs,omitempty"`
}

type ProofCheck struct {
	Name        string   `json:"name"`
	Requirement string   `json:"requirement"`
	Status      string   `json:"status"`
	Evidence    []string `json:"evidence,omitempty"`
	Detail      string   `json:"detail"`
}

type ProofSuiteDetail struct {
	ProofID     string            `json:"proof_id"`
	CartridgeID string            `json:"cartridge_id"`
	Status      string            `json:"status"`
	CreatedAt   string            `json:"created_at"`
	Checks      []ProofCheck      `json:"checks"`
	Artifacts   map[string]string `json:"artifacts"`
	DNABefore   string            `json:"dna_before,omitempty"`
	DNAAfter    string            `json:"dna_after,omitempty"`
	Path        string            `json:"path"`
	HashHop     HashHopReport     `json:"hashhop"`
}
