package store

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"path/filepath"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/config"
	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
	_ "modernc.org/sqlite"
)

type Store struct {
	cfg config.Config
	db  *sql.DB
}

func Open(ctx context.Context, cfg config.Config) (*Store, error) {
	dsn := sqliteReadOnlyDSN(cfg.Database)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, err
	}
	return &Store{cfg: cfg, db: db}, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) Status(ctx context.Context) (domain.Status, error) {
	atomCount, err := s.scalarInt(ctx, "SELECT COUNT(*) FROM atoms")
	if err != nil {
		return domain.Status{}, err
	}
	sourceCount, err := s.scalarInt(ctx, "SELECT COUNT(*) FROM sources")
	if err != nil {
		return domain.Status{}, err
	}
	cartridgeCount, err := s.scalarInt(ctx, "SELECT COUNT(*) FROM cartridges")
	if err != nil {
		return domain.Status{}, err
	}
	activeTokens, err := s.scalarInt64(ctx, "SELECT COALESCE(SUM(token_equivalent), 0) FROM atoms")
	if err != nil {
		return domain.Status{}, err
	}
	cartridgeTokens, err := s.scalarInt64(ctx, "SELECT COALESCE(SUM(token_equivalent), 0) FROM cartridges")
	if err != nil {
		return domain.Status{}, err
	}
	layers, err := s.CacheLayers(ctx)
	if err != nil {
		return domain.Status{}, err
	}
	libraryTokens := activeTokens
	if cartridgeTokens > libraryTokens {
		libraryTokens = cartridgeTokens
	}
	return domain.Status{
		App:                    "Ely-Eye MCP",
		ProjectRoot:            s.cfg.ProjectRoot,
		DataHome:               s.cfg.ElyEyeHome,
		Database:               s.cfg.Database,
		AtomCount:              atomCount,
		SourceCount:            sourceCount,
		CartridgeCount:         cartridgeCount,
		ActiveTokenEquivalent:  activeTokens,
		LibraryTokenEquivalent: libraryTokens,
		CacheLayers:            layers,
	}, nil
}

func (s *Store) CacheLayers(ctx context.Context) ([]domain.CacheLayer, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT
			layer,
			COUNT(DISTINCT cache_key) AS entries,
			COALESCE(SUM(bytes_estimate), 0) AS bytes_estimate,
			SUM(CASE WHEN event = 'hit' THEN 1 ELSE 0 END) AS hit_count,
			SUM(CASE WHEN event = 'miss' THEN 1 ELSE 0 END) AS miss_count,
			COALESCE(MAX(ts), '') AS last_event_at
		FROM cache_events
		GROUP BY layer
		ORDER BY layer`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	layers := []domain.CacheLayer{}
	for rows.Next() {
		var layer domain.CacheLayer
		if err := rows.Scan(&layer.Layer, &layer.Entries, &layer.BytesEstimate, &layer.HitCount, &layer.MissCount, &layer.LastEventAt); err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}
	return layers, rows.Err()
}

func (s *Store) scalarInt(ctx context.Context, query string) (int, error) {
	var value int
	if err := s.db.QueryRowContext(ctx, query).Scan(&value); err != nil {
		return 0, err
	}
	return value, nil
}

func (s *Store) scalarInt64(ctx context.Context, query string) (int64, error) {
	var value int64
	if err := s.db.QueryRowContext(ctx, query).Scan(&value); err != nil {
		return 0, err
	}
	return value, nil
}

func sqliteReadOnlyDSN(path string) string {
	absolute, _ := filepath.Abs(path)
	uri := "file:" + filepath.ToSlash(absolute)
	query := url.Values{}
	query.Set("mode", "ro")
	query.Add("_pragma", "query_only(1)")
	// The production database runs in WAL mode and is written by the Ely-Eye
	// app concurrently. busy_timeout lets a reader wait through a checkpoint or
	// a momentary write lock instead of failing with SQLITE_BUSY. immutable is
	// deliberately not set because the database does change underneath us.
	query.Add("_pragma", "busy_timeout(5000)")
	return fmt.Sprintf("%s?%s", uri, query.Encode())
}
