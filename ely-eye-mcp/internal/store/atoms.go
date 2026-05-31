package store

import (
	"context"
	"database/sql"
	"errors"
	"math"
	"regexp"
	"strings"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
)

var tokenPattern = regexp.MustCompile(`[\p{L}\p{N}_]+`)

func (s *Store) FetchAtom(ctx context.Context, atomID string) (domain.Atom, error) {
	if strings.TrimSpace(atomID) == "" {
		return domain.Atom{}, errors.New("atom_id is required")
	}
	row := s.db.QueryRowContext(ctx, `
		SELECT atom_id, source_id, modality, source, time, text, COALESCE(image_ref, ''),
		       COALESCE(layout_json, ''), relations_json, trust_json, token_equivalent, metadata_json
		FROM atoms
		WHERE atom_id = ?`, atomID)
	atom, err := scanAtom(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return domain.Atom{}, errors.New("atom not found")
		}
		return domain.Atom{}, err
	}
	return atom, nil
}

func (s *Store) SearchAtoms(ctx context.Context, query string, modality string, limit int) ([]domain.AtomHit, error) {
	matchQuery, err := ftsQuery(query)
	if err != nil {
		return nil, err
	}
	if limit <= 0 || limit > 100 {
		limit = 24
	}
	args := []any{matchQuery}
	filter := ""
	if modality != "" {
		filter = "AND a.modality = ?"
		args = append(args, modality)
	}
	args = append(args, limit)
	rows, err := s.db.QueryContext(ctx, `
		SELECT a.atom_id, a.source_id, a.modality, a.source, a.time, a.text,
		       COALESCE(a.image_ref, ''), COALESCE(a.layout_json, ''),
		       a.relations_json, a.trust_json, a.token_equivalent, a.metadata_json,
		       bm25(atom_fts) AS score
		FROM atom_fts
		JOIN atoms a ON a.atom_id = atom_fts.atom_id
		WHERE atom_fts MATCH ? `+filter+`
		ORDER BY score
		LIMIT ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var hits []domain.AtomHit
	var bestScore float64
	for rows.Next() {
		atom, score, err := scanAtomHit(rows)
		if err != nil {
			return nil, err
		}
		sparseScore := -score
		if math.IsNaN(sparseScore) || math.IsInf(sparseScore, 0) {
			sparseScore = 0
		}
		if sparseScore > bestScore {
			bestScore = sparseScore
		}
		hits = append(hits, domain.AtomHit{Atom: atom, SparseScore: sparseScore, FinalScore: sparseScore})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if bestScore > 0 {
		for index := range hits {
			hits[index].SparseScore = hits[index].SparseScore / bestScore
			hits[index].FinalScore = hits[index].SparseScore
		}
	}
	return hits, nil
}

type atomScanner interface {
	Scan(dest ...any) error
}

func scanAtom(scanner atomScanner) (domain.Atom, error) {
	var atom domain.Atom
	var layout string
	var relations string
	var trust string
	var metadata string
	if err := scanner.Scan(
		&atom.AtomID,
		&atom.SourceID,
		&atom.Modality,
		&atom.Source,
		&atom.Time,
		&atom.Text,
		&atom.ImageRef,
		&layout,
		&relations,
		&trust,
		&atom.TokenEquivalent,
		&metadata,
	); err != nil {
		return domain.Atom{}, err
	}
	var err error
	atom.Layout, err = optionalJSONValue(layout)
	if err != nil {
		return domain.Atom{}, err
	}
	atom.Relations, err = requiredJSONValue(relations)
	if err != nil {
		return domain.Atom{}, err
	}
	atom.Trust, err = requiredJSONValue(trust)
	if err != nil {
		return domain.Atom{}, err
	}
	atom.Metadata, err = requiredJSONValue(metadata)
	if err != nil {
		return domain.Atom{}, err
	}
	return atom, nil
}

func scanAtomHit(scanner atomScanner) (domain.Atom, float64, error) {
	var atom domain.Atom
	var layout string
	var relations string
	var trust string
	var metadata string
	var score float64
	if err := scanner.Scan(
		&atom.AtomID,
		&atom.SourceID,
		&atom.Modality,
		&atom.Source,
		&atom.Time,
		&atom.Text,
		&atom.ImageRef,
		&layout,
		&relations,
		&trust,
		&atom.TokenEquivalent,
		&metadata,
		&score,
	); err != nil {
		return domain.Atom{}, 0, err
	}
	var err error
	atom.Layout, err = optionalJSONValue(layout)
	if err != nil {
		return domain.Atom{}, 0, err
	}
	atom.Relations, err = requiredJSONValue(relations)
	if err != nil {
		return domain.Atom{}, 0, err
	}
	atom.Trust, err = requiredJSONValue(trust)
	if err != nil {
		return domain.Atom{}, 0, err
	}
	atom.Metadata, err = requiredJSONValue(metadata)
	if err != nil {
		return domain.Atom{}, 0, err
	}
	return atom, score, nil
}

func ftsQuery(query string) (string, error) {
	tokens := tokenPattern.FindAllString(strings.ToLower(query), -1)
	if len(tokens) == 0 {
		return "", errors.New("query must contain searchable text")
	}
	unique := make([]string, 0, len(tokens))
	seen := map[string]bool{}
	for _, token := range tokens {
		if seen[token] {
			continue
		}
		seen[token] = true
		unique = append(unique, `"`+strings.ReplaceAll(token, `"`, `""`)+`"`)
		if len(unique) == 16 {
			break
		}
	}
	return strings.Join(unique, " OR "), nil
}
