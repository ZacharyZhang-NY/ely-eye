package mcpserver

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/config"
	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/store"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Server struct {
	cfg     config.Config
	store   *store.Store
	version string
}

func Run(ctx context.Context, cfg config.Config, version string) error {
	db, err := store.Open(ctx, cfg)
	if err != nil {
		return err
	}
	defer db.Close()
	server := &Server{cfg: cfg, store: db, version: version}
	mcpServer := mcp.NewServer(&mcp.Implementation{
		Name:    "ely-eye",
		Title:   "Ely-Eye Context Cartridge MCP",
		Version: version,
	}, &mcp.ServerOptions{
		Instructions: "Use Ely-Eye tools to retrieve local cartridge context, evidence atoms, proof suites, and compiled evidence packs. Tools read local Ely-Eye state and do not mutate the production database.",
	})
	server.registerTools(mcpServer)
	server.registerResources(mcpServer)
	if err := mcpServer.Run(ctx, &mcp.StdioTransport{}); err != nil {
		if errors.Is(err, io.EOF) || strings.HasSuffix(err.Error(), "EOF") {
			return nil
		}
		return err
	}
	return nil
}

func (s *Server) registerResources(server *mcp.Server) {
	addFileResource(server, "ely-eye://project/prd", "Ely-Eye PRD", filepath.Join(s.cfg.ProjectRoot, "PRD.md"))
	addFileResource(server, "ely-eye://project/readme", "Ely-Eye README", filepath.Join(s.cfg.ProjectRoot, "README.md"))
	addFileResource(server, "ely-eye://project/mcp-research", "Ely-Eye MCP research notes", filepath.Join(s.cfg.ProjectRoot, "ely-eye-mcp", "docs", "RESEARCH.md"))
}

// addFileResource registers a project document only when it exists on disk.
// External deployments rarely ship the Ely-Eye repository documents, so a
// missing file is a normal condition and the resource is simply omitted.
func addFileResource(server *mcp.Server, uri string, name string, path string) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	size := info.Size()
	server.AddResource(&mcp.Resource{
		URI:         uri,
		Name:        name,
		Title:       name,
		Description: "Local Ely-Eye project resource: " + path,
		MIMEType:    "text/markdown",
		Size:        size,
	}, func(ctx context.Context, request *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
		_ = ctx
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return &mcp.ReadResourceResult{
			Contents: []*mcp.ResourceContents{{
				URI:      request.Params.URI,
				MIMEType: "text/markdown",
				Text:     string(data),
			}},
		}, nil
	})
}
