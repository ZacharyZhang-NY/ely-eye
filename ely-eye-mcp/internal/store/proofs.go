package store

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
)

type proofDir struct {
	name    string
	modTime time.Time
}

func (s *Store) ListProofSuites(ctx context.Context, limit int, includePayload bool) ([]domain.ProofSuite, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	root := filepath.Join(s.cfg.ElyEyeHome, "data", "eval_proofs")
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []domain.ProofSuite{}, nil
		}
		return nil, err
	}
	dirs := sortedProofDirs(entries)
	proofs := make([]domain.ProofSuite, 0, limit)
	for _, dir := range dirs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		path := filepath.Join(root, dir.name, "proof_suite.json")
		data, err := os.ReadFile(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		var header struct {
			ProofID     string `json:"proof_id"`
			CartridgeID string `json:"cartridge_id"`
			Status      string `json:"status"`
			CreatedAt   string `json:"created_at"`
		}
		if err := json.Unmarshal(data, &header); err != nil {
			return nil, err
		}
		proof := domain.ProofSuite{
			ProofID:     header.ProofID,
			CartridgeID: header.CartridgeID,
			Status:      header.Status,
			CreatedAt:   header.CreatedAt,
			Path:        path,
		}
		if includePayload {
			value, err := readJSONValue(data)
			if err != nil {
				return nil, err
			}
			proof.Payload = value
		}
		proofs = append(proofs, proof)
		if len(proofs) == limit {
			break
		}
	}
	return proofs, nil
}

// sortedProofDirs resolves each directory's mod time once and orders newest
// first. The store mutates underneath the read-only server, so a directory may
// be removed between ReadDir and Info; an Info error yields a zero time that
// sorts last rather than dereferencing a nil FileInfo. Names break ties for a
// stable order.
func sortedProofDirs(entries []os.DirEntry) []proofDir {
	dirs := make([]proofDir, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		modTime := time.Time{}
		if info, err := entry.Info(); err == nil {
			modTime = info.ModTime()
		}
		dirs = append(dirs, proofDir{name: entry.Name(), modTime: modTime})
	}
	sort.Slice(dirs, func(i int, j int) bool {
		if !dirs[i].modTime.Equal(dirs[j].modTime) {
			return dirs[i].modTime.After(dirs[j].modTime)
		}
		return dirs[i].name < dirs[j].name
	})
	return dirs
}
