package mcpserver

import (
	"context"
	"errors"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/contextpack"
	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type EmptyInput struct{}

type ListCartridgesInput struct {
	Limit int `json:"limit,omitempty" jsonschema:"maximum number of cartridges to return, capped at 200"`
}

type GetCartridgeInput struct {
	CartridgeID string `json:"cartridge_id,omitempty" jsonschema:"exact Ely-Eye cartridge id"`
	Name        string `json:"name,omitempty" jsonschema:"exact Ely-Eye cartridge name"`
}

type SearchAtomsInput struct {
	Query    string `json:"query" jsonschema:"search query for the Ely-Eye FTS atom index"`
	Modality string `json:"modality,omitempty" jsonschema:"optional exact modality filter such as text, code, image, pdf_page, pdf_region, video_frame, ui_screenshot"`
	Limit    int    `json:"limit,omitempty" jsonschema:"maximum number of atoms to return, capped at 100"`
}

type FetchAtomInput struct {
	AtomID string `json:"atom_id" jsonschema:"exact Evidence Atom id"`
}

type CompileContextInput struct {
	Question    string `json:"question" jsonschema:"question to compile into an Ely-Eye evidence pack"`
	Profile     string `json:"profile,omitempty" jsonschema:"live_demo, extreme_context, library_100m, or research_theater"`
	Limit       int    `json:"limit,omitempty" jsonschema:"maximum retrieval hits to pack"`
	TokenBudget int64  `json:"token_budget,omitempty" jsonschema:"explicit token budget for the evidence pack"`
}

type ListProofSuitesInput struct {
	Limit          int  `json:"limit,omitempty" jsonschema:"maximum proof suites to return, capped at 100"`
	IncludePayload bool `json:"include_payload,omitempty" jsonschema:"include full proof_suite.json payloads"`
}

type ListHashHopProofsInput struct {
	Kind  string `json:"kind,omitempty" jsonschema:"optional filter: text_hashhop or visual_hashhop"`
	Limit int    `json:"limit,omitempty" jsonschema:"maximum example proofs to include, capped at 200"`
}

type GetProofSuiteInput struct {
	ProofID string `json:"proof_id" jsonschema:"exact proof suite id, for example prd_proof_suite_6bc3fa744c60a0e6"`
}

type StatusOutput struct {
	Status domain.Status `json:"status"`
}

type ListCartridgesOutput struct {
	Cartridges []domain.Cartridge `json:"cartridges"`
}

type GetCartridgeOutput struct {
	Cartridge domain.CartridgeDetail `json:"cartridge"`
}

type SearchAtomsOutput struct {
	Hits []domain.AtomSearchHit `json:"hits"`
}

type FetchAtomOutput struct {
	Atom domain.Atom `json:"atom"`
}

type CompileContextOutput struct {
	Context domain.EvidencePack `json:"context"`
}

type ListProofSuitesOutput struct {
	ProofSuites []domain.ProofSuite `json:"proof_suites"`
}

type ListHashHopProofsOutput struct {
	Report domain.HashHopReport `json:"report"`
}

type GetProofSuiteOutput struct {
	ProofSuite domain.ProofSuiteDetail `json:"proof_suite"`
}

func (s *Server) registerTools(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{
		Name:        "ely_eye_status",
		Title:       "Ely-Eye status",
		Description: "Return read-only counts and cache layer status from the local Ely-Eye database.",
	}, s.statusTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_cartridges",
		Title:       "List Context Cartridges",
		Description: "List local Ely-Eye Context Cartridges with DNA, token-equivalent capacity, and manifest JSON.",
	}, s.listCartridgesTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "get_cartridge",
		Title:       "Get Context Cartridge",
		Description: "Read one local Ely-Eye Context Cartridge, including artifacts, memory capsule index, and asset report when present.",
	}, s.getCartridgeTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search_atoms",
		Title:       "Search Evidence Atoms",
		Description: "Search the local Ely-Eye FTS index and return ranked Evidence Atoms.",
	}, s.searchAtomsTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch_atom",
		Title:       "Fetch Evidence Atom",
		Description: "Read one Evidence Atom by exact atom id.",
	}, s.fetchAtomTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "compile_context",
		Title:       "Compile Evidence Context",
		Description: "Compile a question into an Ely-Eye evidence pack using local retrieval and the same profile names as the app.",
	}, s.compileContextTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_proof_suites",
		Title:       "List Proof Suites",
		Description: "List local PRD proof suites from .ely_eye/data/eval_proofs.",
	}, s.listProofSuitesTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_hashhop_proofs",
		Title:       "List HashHop Proofs",
		Description: "Aggregate local HashHop and Visual HashHop long-context addressing proofs by kind, hop count, and token budget, with pass counts and example proofs.",
	}, s.listHashHopProofsTool)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "get_proof_suite",
		Title:       "Get Proof Suite",
		Description: "Read one local PRD proof suite by id, including all checks and a summary of its HashHop proofs.",
	}, s.getProofSuiteTool)
}

func (s *Server) statusTool(ctx context.Context, _ *mcp.CallToolRequest, _ EmptyInput) (*mcp.CallToolResult, StatusOutput, error) {
	status, err := s.store.Status(ctx)
	if err != nil {
		return nil, StatusOutput{}, err
	}
	return nil, StatusOutput{Status: status}, nil
}

func (s *Server) listCartridgesTool(ctx context.Context, _ *mcp.CallToolRequest, input ListCartridgesInput) (*mcp.CallToolResult, ListCartridgesOutput, error) {
	cartridges, err := s.store.ListCartridges(ctx, input.Limit)
	if err != nil {
		return nil, ListCartridgesOutput{}, err
	}
	return nil, ListCartridgesOutput{Cartridges: cartridges}, nil
}

func (s *Server) getCartridgeTool(ctx context.Context, _ *mcp.CallToolRequest, input GetCartridgeInput) (*mcp.CallToolResult, GetCartridgeOutput, error) {
	cartridge, err := s.store.GetCartridge(ctx, input.CartridgeID, input.Name)
	if err != nil {
		return nil, GetCartridgeOutput{}, err
	}
	return nil, GetCartridgeOutput{Cartridge: cartridge}, nil
}

func (s *Server) searchAtomsTool(ctx context.Context, _ *mcp.CallToolRequest, input SearchAtomsInput) (*mcp.CallToolResult, SearchAtomsOutput, error) {
	hits, err := s.store.SearchAtoms(ctx, input.Query, input.Modality, input.Limit)
	if err != nil {
		return nil, SearchAtomsOutput{}, err
	}
	return nil, SearchAtomsOutput{Hits: contextpack.SummarizeHits(hits, 1200)}, nil
}

func (s *Server) fetchAtomTool(ctx context.Context, _ *mcp.CallToolRequest, input FetchAtomInput) (*mcp.CallToolResult, FetchAtomOutput, error) {
	atom, err := s.store.FetchAtom(ctx, input.AtomID)
	if err != nil {
		return nil, FetchAtomOutput{}, err
	}
	return nil, FetchAtomOutput{Atom: atom}, nil
}

func (s *Server) compileContextTool(ctx context.Context, _ *mcp.CallToolRequest, input CompileContextInput) (*mcp.CallToolResult, CompileContextOutput, error) {
	if input.Question == "" {
		return nil, CompileContextOutput{}, errors.New("question is required")
	}
	hits, err := s.store.SearchAtoms(ctx, input.Question, "", input.Limit)
	if err != nil {
		return nil, CompileContextOutput{}, err
	}
	pack := contextpack.Build(input.Question, input.Profile, hits, input.TokenBudget)
	return nil, CompileContextOutput{Context: pack}, nil
}

func (s *Server) listProofSuitesTool(ctx context.Context, _ *mcp.CallToolRequest, input ListProofSuitesInput) (*mcp.CallToolResult, ListProofSuitesOutput, error) {
	proofs, err := s.store.ListProofSuites(ctx, input.Limit, input.IncludePayload)
	if err != nil {
		return nil, ListProofSuitesOutput{}, err
	}
	return nil, ListProofSuitesOutput{ProofSuites: proofs}, nil
}

func (s *Server) listHashHopProofsTool(ctx context.Context, _ *mcp.CallToolRequest, input ListHashHopProofsInput) (*mcp.CallToolResult, ListHashHopProofsOutput, error) {
	report, err := s.store.ListHashHopProofs(ctx, input.Kind, input.Limit)
	if err != nil {
		return nil, ListHashHopProofsOutput{}, err
	}
	return nil, ListHashHopProofsOutput{Report: report}, nil
}

func (s *Server) getProofSuiteTool(ctx context.Context, _ *mcp.CallToolRequest, input GetProofSuiteInput) (*mcp.CallToolResult, GetProofSuiteOutput, error) {
	suite, err := s.store.GetProofSuite(ctx, input.ProofID)
	if err != nil {
		return nil, GetProofSuiteOutput{}, err
	}
	return nil, GetProofSuiteOutput{ProofSuite: suite}, nil
}
