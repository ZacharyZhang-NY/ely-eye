package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/cli"
)

func main() {
	if err := cli.Run(context.Background(), os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
