package store

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
)

func (s *Store) evalProofsRoot() string {
	return filepath.Join(s.cfg.ElyEyeHome, "data", "eval_proofs")
}

// ListHashHopProofs aggregates every HashHop and Visual HashHop proof under the
// eval set into pass-rate buckets keyed by kind, hop count, and token budget,
// plus a capped list of recent example proofs.
func (s *Store) ListHashHopProofs(ctx context.Context, kind string, limit int) (domain.HashHopReport, error) {
	if kind != "" && kind != "text_hashhop" && kind != "visual_hashhop" {
		return domain.HashHopReport{}, errors.New("kind must be text_hashhop or visual_hashhop")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}
	return aggregateHashHop(ctx, s.evalProofsRoot(), kind, limit)
}

// GetProofSuite reads one PRD proof suite by id, including all checks, and
// summarizes the HashHop proofs stored alongside it.
func (s *Store) GetProofSuite(ctx context.Context, proofID string) (domain.ProofSuiteDetail, error) {
	if err := validateProofID(proofID); err != nil {
		return domain.ProofSuiteDetail{}, err
	}
	dir := filepath.Join(s.evalProofsRoot(), proofID)
	path := filepath.Join(dir, "proof_suite.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return domain.ProofSuiteDetail{}, errors.New("proof suite not found")
		}
		return domain.ProofSuiteDetail{}, err
	}
	var detail domain.ProofSuiteDetail
	if err := json.Unmarshal(data, &detail); err != nil {
		return domain.ProofSuiteDetail{}, err
	}
	detail.Path = path
	report, err := aggregateHashHop(ctx, filepath.Join(dir, "hashhop"), "", 0)
	if err != nil {
		return domain.ProofSuiteDetail{}, err
	}
	detail.HashHop = report
	return detail, nil
}

// aggregateHashHop walks a directory tree for proof.json files and builds a
// HashHopReport. proofCap <= 0 returns buckets only; a positive cap also
// includes that many most-recent example proofs.
func aggregateHashHop(ctx context.Context, root string, kind string, proofCap int) (domain.HashHopReport, error) {
	buckets := map[string]*domain.HashHopBucket{}
	seen := map[string]bool{}
	var proofs []domain.HashHopProof
	report := domain.HashHopReport{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() || entry.Name() != "proof.json" {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var proof domain.HashHopProof
		if err := json.Unmarshal(data, &proof); err != nil {
			return err
		}
		if kind != "" && proof.Kind != kind {
			return nil
		}
		// A suite stores each HashHop proof under its own hashhop/ directory and
		// also promotes a byte-identical copy to a top-level proof directory.
		// Count each proof once by its content-derived id.
		if seen[proof.ProofID] {
			return nil
		}
		seen[proof.ProofID] = true
		proof.Path = path
		report.Total++
		passed := proof.Passed != nil && *proof.Passed
		if passed {
			report.Passed++
		}
		key := proof.Kind + "|" + strconv.Itoa(proof.Hops) + "|" + strconv.FormatInt(proof.TokenEquivalent, 10)
		bucket, ok := buckets[key]
		if !ok {
			bucket = &domain.HashHopBucket{Kind: proof.Kind, Hops: proof.Hops, TokenEquivalent: proof.TokenEquivalent}
			buckets[key] = bucket
		}
		bucket.Total++
		if passed {
			bucket.Passed++
		}
		if proofCap > 0 {
			proofs = append(proofs, proof)
		}
		return nil
	})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return domain.HashHopReport{Buckets: []domain.HashHopBucket{}}, nil
		}
		return domain.HashHopReport{}, err
	}
	report.Buckets = sortedBuckets(buckets)
	if proofCap > 0 {
		report.Proofs = capRecentProofs(proofs, proofCap)
	}
	return report, nil
}

func sortedBuckets(buckets map[string]*domain.HashHopBucket) []domain.HashHopBucket {
	out := make([]domain.HashHopBucket, 0, len(buckets))
	for _, bucket := range buckets {
		out = append(out, *bucket)
	}
	sort.Slice(out, func(i int, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].TokenEquivalent != out[j].TokenEquivalent {
			return out[i].TokenEquivalent < out[j].TokenEquivalent
		}
		return out[i].Hops < out[j].Hops
	})
	return out
}

func capRecentProofs(proofs []domain.HashHopProof, limit int) []domain.HashHopProof {
	sort.Slice(proofs, func(i int, j int) bool {
		return proofs[i].CreatedAt > proofs[j].CreatedAt
	})
	if len(proofs) > limit {
		proofs = proofs[:limit]
	}
	return proofs
}

func validateProofID(id string) error {
	if id == "" {
		return errors.New("proof_id is required")
	}
	if id == "." || id == ".." || strings.ContainsAny(id, `/\`) || filepath.VolumeName(id) != "" || filepath.Base(id) != id {
		return errors.New("proof_id must be a single path element")
	}
	return nil
}
