package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
)

func (s *Store) ListCartridges(ctx context.Context, limit int) ([]domain.Cartridge, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT cartridge_id, name, root_path, manifest_json, COALESCE(dna, ''),
		       token_equivalent, created_at
		FROM cartridges
		ORDER BY created_at DESC
		LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	cartridges := []domain.Cartridge{}
	for rows.Next() {
		cartridge, err := scanCartridge(rows)
		if err != nil {
			return nil, err
		}
		cartridges = append(cartridges, cartridge)
	}
	return cartridges, rows.Err()
}

func (s *Store) GetCartridge(ctx context.Context, cartridgeID string, name string) (domain.CartridgeDetail, error) {
	if cartridgeID == "" && name == "" {
		return domain.CartridgeDetail{}, errors.New("cartridge_id or name is required")
	}
	query := `
		SELECT cartridge_id, name, root_path, manifest_json, COALESCE(dna, ''),
		       token_equivalent, created_at
		FROM cartridges
		WHERE cartridge_id = ? OR name = ?
		ORDER BY created_at DESC
		LIMIT 1`
	row := s.db.QueryRowContext(ctx, query, cartridgeID, name)
	cartridge, err := scanCartridge(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return domain.CartridgeDetail{}, errors.New("cartridge not found")
		}
		return domain.CartridgeDetail{}, err
	}
	artifacts, err := artifactsFromManifest(cartridge.Manifest)
	if err != nil {
		return domain.CartridgeDetail{}, err
	}
	detail := domain.CartridgeDetail{
		Cartridge: cartridge,
		Artifacts: artifacts,
	}
	memoryIndex, err := readOptionalJSON(filepath.Join(cartridge.RootPath, "memory_capsule_index.json"))
	if err != nil {
		return domain.CartridgeDetail{}, err
	}
	detail.MemoryCapsuleIndex = memoryIndex
	assetReport, err := readOptionalJSON(filepath.Join(cartridge.RootPath, "cartridge_assets.json"))
	if err != nil {
		return domain.CartridgeDetail{}, err
	}
	detail.AssetReport = assetReport
	return detail, nil
}

type cartridgeScanner interface {
	Scan(dest ...any) error
}

func scanCartridge(scanner cartridgeScanner) (domain.Cartridge, error) {
	var cartridge domain.Cartridge
	var manifest string
	if err := scanner.Scan(
		&cartridge.CartridgeID,
		&cartridge.Name,
		&cartridge.RootPath,
		&manifest,
		&cartridge.DNA,
		&cartridge.TokenEquivalent,
		&cartridge.CreatedAt,
	); err != nil {
		return domain.Cartridge{}, err
	}
	value, err := requiredJSONValue(manifest)
	if err != nil {
		return domain.Cartridge{}, err
	}
	cartridge.Manifest = value
	return cartridge, nil
}

func artifactsFromManifest(manifest any) (map[string]string, error) {
	var payload struct {
		Artifacts map[string]string `json:"artifacts"`
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	if payload.Artifacts == nil {
		payload.Artifacts = map[string]string{}
	}
	return payload.Artifacts, nil
}

func readOptionalJSON(path string) (any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	value, err := readJSONValue(data)
	if err != nil {
		return nil, errors.New(err.Error() + " at " + path)
	}
	return value, nil
}
