package dataextract

import (
	"bytes"
	"strings"
	"testing"
)

func TestExtractSimpleCSVNoHeader(t *testing.T) {
	in := strings.NewReader("a,b,c\n1,2,3\n")
	var out bytes.Buffer
	if err := ExtractSimpleCSV(in, &out, SimpleOptions{HasHeader: false}); err != nil {
		t.Fatalf("extract simple csv: %v", err)
	}
	if got := out.String(); got != "col0,col1,col2\na,b,c\n1,2,3\n" {
		t.Fatalf("unexpected simple output:\n%s", got)
	}
}

func TestExtractVowelRecognitionCSV(t *testing.T) {
	in := strings.NewReader("1,0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1\n")
	var out bytes.Buffer
	if err := ExtractVowelRecognitionCSV(in, &out, VowelRecognitionOptions{HasHeader: false}); err != nil {
		t.Fatalf("extract vowel csv: %v", err)
	}
	got := out.String()
	if !strings.HasPrefix(got, "type0,type1,type2,feature0,feature1") {
		t.Fatalf("unexpected vowel header:\n%s", got)
	}
	if !strings.Contains(got, "\n1,0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1\n") {
		t.Fatalf("unexpected vowel row:\n%s", got)
	}
}

func TestExtractABCPred1CSV(t *testing.T) {
	in := strings.NewReader("SEQAAA,1\nSEQBBB,0\n")
	var out bytes.Buffer
	if err := ExtractABCPred1CSV(in, &out, ABCPred1Options{HasHeader: false}); err != nil {
		t.Fatalf("extract abc_pred1 csv: %v", err)
	}
	if got := out.String(); got != "sequence,class\nSEQAAA,1\nSEQBBB,0\n" {
		t.Fatalf("unexpected abc_pred1 output:\n%s", got)
	}
}

func TestExtractHedgeFundCSV(t *testing.T) {
	in := strings.NewReader("date,a,b,c,last\n2020-01-01,1,2,3,x\n")
	var out bytes.Buffer
	if err := ExtractHedgeFundCSV(in, &out, HedgeFundOptions{HasHeader: true}); err != nil {
		t.Fatalf("extract hedge_fund csv: %v", err)
	}
	if got := out.String(); got != "a,b,c,date\n1,2,3,2020-01-01\n" {
		t.Fatalf("unexpected hedge_fund output:\n%s", got)
	}
}

func TestExtractMinesVsRocksCSV(t *testing.T) {
	in := strings.NewReader("*CM001,0.1,0.2\nCR001,0.3,0.4\nUNKNOWN,0.5,0.6\n")
	var out bytes.Buffer
	if err := ExtractMinesVsRocksCSV(in, &out, MinesVsRocksOptions{HasHeader: false}); err != nil {
		t.Fatalf("extract mines_vs_rocks csv: %v", err)
	}
	if got := out.String(); got != "split,feature0,feature1,class0,class1\n0,0.1,0.2,1,0\n1,0.3,0.4,0,1\n" {
		t.Fatalf("unexpected mines_vs_rocks output:\n%s", got)
	}
}

func TestExtractChromHMMExpandedCSV(t *testing.T) {
	in := strings.NewReader("chr,from,to,tag\nchr22,100,500,Enh\n")
	var out bytes.Buffer
	if err := ExtractChromHMMExpandedCSV(in, &out, ChromHMMExpandedOptions{
		HasHeader: true,
		Step:      200,
	}); err != nil {
		t.Fatalf("extract chrom_hmm expanded csv: %v", err)
	}
	got := out.String()
	if !strings.HasPrefix(got, "tag_Enh,bp_index,tag,chrom\n") {
		t.Fatalf("unexpected chrom_hmm header:\n%s", got)
	}
	if !strings.Contains(got, "1,100,Enh,chr22\n") {
		t.Fatalf("missing first expanded row:\n%s", got)
	}
	if !strings.Contains(got, "1,500,Enh,chr22\n") {
		t.Fatalf("missing last expanded row:\n%s", got)
	}
}

func TestChromHMMKnownTags(t *testing.T) {
	tags := ChromHMMKnownTags()
	if len(tags) != 25 {
		t.Fatalf("unexpected tag count: %d", len(tags))
	}
	if tags[0] != "ReprD" || tags[len(tags)-1] != "Quies" {
		t.Fatalf("unexpected known tag ordering: %v", tags)
	}
}
