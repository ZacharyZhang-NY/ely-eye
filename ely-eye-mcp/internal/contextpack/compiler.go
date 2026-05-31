package contextpack

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/ZacharyZhang-NY/ely-eye/ely-eye-mcp/internal/domain"
)

var profileBudgets = map[string]int64{
	"live_demo":        128000,
	"extreme_context":  1010000,
	"library_100m":     262144,
	"research_theater": 128000,
}

func Build(question string, profile string, hits []domain.AtomHit, requestedBudget int64) domain.EvidencePack {
	if profile == "" {
		profile = "library_100m"
	}
	budget := requestedBudget
	if budget <= 0 {
		budget = profileBudgets[profile]
	}
	if budget <= 0 {
		budget = profileBudgets["library_100m"]
	}
	var sections []string
	var tokens int64
	for _, hit := range hits {
		text := focusText(hit.Atom.Text, question, 1600)
		atomTokens := tokenEquivalent(text)
		if atomTokens == 0 {
			atomTokens = 1
		}
		if tokens+atomTokens > budget {
			break
		}
		sections = append(sections, evidenceSection(hit, text))
		tokens += atomTokens
	}
	trace := traceID(question, profile, tokens, hits)
	return domain.EvidencePack{
		Question:        question,
		Profile:         profile,
		TokenBudget:     budget,
		TokenEquivalent: tokens,
		CacheTraceID:    trace,
		Hits:            SummarizeHits(hits, 900),
		PackedText:      strings.Join(sections, "\n\n"),
		VerifierContract: []string{
			"Every factual claim must cite atom ids.",
			"Citations must include page, image region, frame, or code path when available.",
			"Contradictions must be reported as contradiction_notes.",
		},
	}
}

func SummarizeHits(hits []domain.AtomHit, maxRunes int) []domain.AtomSearchHit {
	summaries := make([]domain.AtomSearchHit, 0, len(hits))
	for _, hit := range hits {
		summaries = append(summaries, domain.AtomSearchHit{
			Atom:        summarizeAtom(hit.Atom, maxRunes),
			SparseScore: hit.SparseScore,
			FinalScore:  hit.FinalScore,
		})
	}
	return summaries
}

func summarizeAtom(atom domain.Atom, maxRunes int) domain.AtomSummary {
	return domain.AtomSummary{
		AtomID:          atom.AtomID,
		SourceID:        atom.SourceID,
		Modality:        atom.Modality,
		Source:          atom.Source,
		Time:            atom.Time,
		TextPreview:     trimRunes(atom.Text, maxRunes),
		ImageRef:        atom.ImageRef,
		TokenEquivalent: atom.TokenEquivalent,
	}
}

func evidenceSection(hit domain.AtomHit, text string) string {
	return fmt.Sprintf(
		"[EVIDENCE]\natom_id=%s\nmodality=%s\nsource=%s\nimage_ref=%s\nscore=%.4f\n%s\n[/EVIDENCE]",
		hit.Atom.AtomID,
		hit.Atom.Modality,
		hit.Atom.Source,
		hit.Atom.ImageRef,
		hit.FinalScore,
		text,
	)
}

func focusText(text string, query string, window int) string {
	if utf8.RuneCountInString(text) <= window {
		return text
	}
	lowerText := strings.ToLower(text)
	best := 0
	for _, term := range strings.Fields(strings.ToLower(query)) {
		if len(term) < 3 {
			continue
		}
		if index := strings.Index(lowerText, term); index >= 0 {
			best = index
			break
		}
	}
	start := best - window/2
	if start < 0 {
		start = 0
	}
	runes := []rune(text)
	runeStart := byteToRuneOffset(text, start)
	if runeStart+window > len(runes) {
		runeStart = len(runes) - window
	}
	if runeStart < 0 {
		runeStart = 0
	}
	return string(runes[runeStart : runeStart+window])
}

func trimRunes(text string, limit int) string {
	if limit <= 0 || utf8.RuneCountInString(text) <= limit {
		return text
	}
	runes := []rune(text)
	return string(runes[:limit])
}

func tokenEquivalent(text string) int64 {
	var ascii int64
	var nonASCII int64
	for _, r := range text {
		if r < 128 {
			ascii++
		} else {
			nonASCII++
		}
	}
	total := ascii/4 + nonASCII/2
	if total == 0 && text != "" {
		return 1
	}
	return total
}

func traceID(question string, profile string, tokens int64, hits []domain.AtomHit) string {
	hash := sha256.New()
	fmt.Fprintf(hash, "%s\n%s\n%d\n", question, profile, tokens)
	for _, hit := range hits {
		fmt.Fprintln(hash, hit.Atom.AtomID)
	}
	return hex.EncodeToString(hash.Sum(nil))[:16]
}

func byteToRuneOffset(text string, byteOffset int) int {
	if byteOffset <= 0 {
		return 0
	}
	count := 0
	for index := range text {
		if index >= byteOffset {
			return count
		}
		count++
	}
	return count
}
