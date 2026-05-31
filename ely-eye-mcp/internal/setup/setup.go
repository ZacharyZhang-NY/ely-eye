package setup

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/config"
)

type Options struct {
	Client     string
	Scope      string
	ServerName string
}

type Action struct {
	Client  string `json:"client"`
	Scope   string `json:"scope"`
	Target  string `json:"target"`
	Detail  string `json:"detail"`
	Command string `json:"command,omitempty"`
}

type Result struct {
	ServerName  string   `json:"server_name"`
	Binary      string   `json:"binary"`
	ProjectRoot string   `json:"project_root"`
	Actions     []Action `json:"actions"`
}

func Run(ctx context.Context, cfg config.Config, options Options) (Result, error) {
	client := normalizeClient(options.Client)
	if client != "codex" && client != "claude" && client != "both" {
		return Result{}, errors.New("client must be codex, claude, or both")
	}
	scope := normalizeScope(options.Scope)
	if scope != "project" && scope != "user" {
		return Result{}, errors.New("claude scope must be project or user")
	}
	serverName := options.ServerName
	if serverName == "" {
		serverName = "ely-eye"
	}
	if cfg.Binary == "" {
		return Result{}, errors.New("binary path is required for setup")
	}
	result := Result{ServerName: serverName, Binary: cfg.Binary, ProjectRoot: cfg.ProjectRoot}
	var actions []Action
	if client == "codex" || client == "both" {
		action, err := setupCodex(ctx, cfg, serverName)
		if err != nil {
			return Result{}, err
		}
		actions = append(actions, action)
	}
	if client == "claude" || client == "both" {
		action, err := setupClaude(ctx, cfg, serverName, scope)
		if err != nil {
			return Result{}, err
		}
		actions = append(actions, action)
	}
	result.Actions = actions
	return result, nil
}

func normalizeClient(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "all", "both":
		return "both"
	case "codex", "claude":
		return strings.ToLower(value)
	default:
		return value
	}
}

func normalizeScope(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "project":
		return "project"
	case "user":
		return "user"
	default:
		return value
	}
}

func serverCommand(cfg config.Config) []string {
	return []string{
		cfg.Binary,
		"server",
		"--project-root",
		cfg.ProjectRoot,
		"--ely-eye-home",
		cfg.ElyEyeHome,
	}
}

func serverEnv(cfg config.Config) map[string]string {
	return map[string]string{
		"ELY_EYE_HOME": cfg.ElyEyeHome,
	}
}

func commandLine(parts []string) string {
	quoted := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.ContainsAny(part, " \t\"'") {
			quoted = append(quoted, `"`+strings.ReplaceAll(part, `"`, `\"`)+`"`)
		} else {
			quoted = append(quoted, part)
		}
	}
	return strings.Join(quoted, " ")
}

func setupCodex(ctx context.Context, cfg config.Config, serverName string) (Action, error) {
	codex, err := exec.LookPath("codex")
	if err != nil {
		return Action{}, errors.New("codex CLI is required to configure Codex MCP")
	}
	remove := exec.CommandContext(ctx, codex, "mcp", "remove", serverName)
	_ = remove.Run()
	args := []string{"mcp", "add", serverName, "--env", "ELY_EYE_HOME=" + cfg.ElyEyeHome, "--"}
	args = append(args, serverCommand(cfg)...)
	add := exec.CommandContext(ctx, codex, args...)
	output, err := add.CombinedOutput()
	if err != nil {
		return Action{}, errors.New("codex mcp add failed: " + strings.TrimSpace(string(output)))
	}
	return Action{
		Client:  "codex",
		Scope:   "user",
		Target:  "~/.codex/config.toml",
		Detail:  "registered stdio MCP server through Codex CLI",
		Command: commandLine(append([]string{"codex"}, args...)),
	}, nil
}

func setupClaude(ctx context.Context, cfg config.Config, serverName string, scope string) (Action, error) {
	if scope == "user" {
		return setupClaudeUser(ctx, cfg, serverName)
	}
	if scope != "project" {
		return Action{}, errors.New("claude scope must be project or user")
	}
	target := filepath.Join(cfg.ProjectRoot, ".mcp.json")
	if err := writeProjectMCP(target, serverName, cfg); err != nil {
		return Action{}, err
	}
	return Action{
		Client: "claude",
		Scope:  "project",
		Target: target,
		Detail: "wrote project-scoped .mcp.json",
	}, nil
}

func setupClaudeUser(ctx context.Context, cfg config.Config, serverName string) (Action, error) {
	claude, err := exec.LookPath("claude")
	if err != nil {
		return Action{}, errors.New("claude CLI is required for user-scoped Claude Code MCP setup")
	}
	payload, err := json.Marshal(mcpServerConfig(cfg))
	if err != nil {
		return Action{}, err
	}
	args := []string{"mcp", "add-json", serverName, string(payload), "--scope", "user"}
	cmd := exec.CommandContext(ctx, claude, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return Action{}, errors.New("claude mcp add-json failed: " + strings.TrimSpace(string(output)))
	}
	return Action{
		Client:  "claude",
		Scope:   "user",
		Target:  "~/.claude.json",
		Detail:  "registered stdio MCP server through Claude CLI",
		Command: commandLine(append([]string{"claude"}, args...)),
	}, nil
}

func writeProjectMCP(path string, serverName string, cfg config.Config) error {
	payload := map[string]any{}
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &payload); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	servers, ok := payload["mcpServers"].(map[string]any)
	if !ok {
		servers = map[string]any{}
	}
	servers[serverName] = mcpServerConfig(cfg)
	payload["mcpServers"] = servers
	data, err = json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}

func mcpServerConfig(cfg config.Config) map[string]any {
	command := serverCommand(cfg)
	return map[string]any{
		"type":    "stdio",
		"command": command[0],
		"args":    command[1:],
		"env":     serverEnv(cfg),
	}
}
