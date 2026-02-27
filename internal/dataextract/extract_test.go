package dataextract

import (
	"strings"
	"testing"
)

func TestExtractSeriesCSVByHeaderName(t *testing.T) {
	in := strings.NewReader("t,close,volume\n0,1.01,10\n1,1.02,11\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,1.01\n1,1.02\n" {
		t.Fatalf("unexpected series output:\n%s", got)
	}
}

func TestExtractEpitopesCSVWithNamedSequenceColumns(t *testing.T) {
	in := strings.NewReader("signal,memory,class,aa0,aa1\n0.1,0.0,1,4,10\n-0.2,0.1,0,2,2\n")
	var out strings.Builder
	err := ExtractEpitopesCSV(in, &out, EpitopesOptions{
		HasHeader:           true,
		SignalColumnName:    "signal",
		MemoryColumnName:    "memory",
		ClassColumnName:     "class",
		SequenceColumnNames: []string{"aa0", "aa1"},
		SignalColumnIndex:   -1,
		MemoryColumnIndex:   -1,
		ClassColumnIndex:    -1,
	})
	if err != nil {
		t.Fatalf("extract epitopes: %v", err)
	}
	if got := out.String(); got != "signal,memory,class,seq0,seq1\n0.1,0,1,4,10\n-0.2,0.1,0,2,2\n" {
		t.Fatalf("unexpected epitopes output:\n%s", got)
	}
}

func TestExtractSeriesCSVNormalizeMinMax(t *testing.T) {
	in := strings.NewReader("close\n10\n20\n30\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
		Normalize:         "minmax",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,0\n1,0.5\n2,1\n" {
		t.Fatalf("unexpected minmax output:\n%s", got)
	}
}

func TestExtractSeriesCSVNormalizeZScore(t *testing.T) {
	in := strings.NewReader("close\n10\n20\n30\n")
	var out strings.Builder
	err := ExtractSeriesCSV(in, &out, SeriesOptions{
		HasHeader:         true,
		ValueColumnName:   "close",
		ValueColumnIndex:  -1,
		OutputValueHeader: "value",
		Normalize:         "zscore",
	})
	if err != nil {
		t.Fatalf("extract series: %v", err)
	}
	if got := out.String(); got != "t,value\n0,-1.224744871391589\n1,0\n2,1.224744871391589\n" {
		t.Fatalf("unexpected zscore output:\n%s", got)
	}
}

func TestExtractMNISTCSVDefaultClassLastOneHot(t *testing.T) {
	in := strings.NewReader("px0,px1,label\n0,255,0\n128,64,9\n")
	var out strings.Builder
	err := ExtractMNISTCSV(in, &out, MNISTOptions{
		HasHeader:            true,
		LabelColumnName:      "label",
		LabelColumnIndex:     -1,
		OneHotClassification: true,
	})
	if err != nil {
		t.Fatalf("extract mnist: %v", err)
	}
	want := "px0,px1,class9,class8,class7,class6,class5,class4,class3,class2,class1,class0\n" +
		"0,255,0,0,0,0,0,0,0,0,0,1\n" +
		"128,64,1,0,0,0,0,0,0,0,0,0\n"
	if got := out.String(); got != want {
		t.Fatalf("unexpected mnist output:\n%s", got)
	}
}

func TestExtractWineCSVOneHotParityMapping(t *testing.T) {
	in := strings.NewReader("class,f0,f1\n1,10.1,11.2\n2,12.3,13.4\n3,14.5,15.6\n")
	var out strings.Builder
	err := ExtractWineCSV(in, &out, WineOptions{
		HasHeader:            true,
		LabelColumnName:      "class",
		LabelColumnIndex:     -1,
		OneHotClassification: true,
	})
	if err != nil {
		t.Fatalf("extract wine: %v", err)
	}
	want := "f0,f1,class3,class2,class1\n" +
		"10.1,11.2,0,0,1\n" +
		"12.3,13.4,0,1,0\n" +
		"14.5,15.6,1,0,0\n"
	if got := out.String(); got != want {
		t.Fatalf("unexpected wine output:\n%s", got)
	}
}

func TestExtractChrHMMCSVDefaultColumns(t *testing.T) {
	in := strings.NewReader("chr,from,to,tag,a,b\nchr22,100,200,Enh,x,y\n")
	var out strings.Builder
	err := ExtractChrHMMCSV(in, &out, ChrHMMOptions{
		HasHeader: true,
	})
	if err != nil {
		t.Fatalf("extract chr_hmm: %v", err)
	}
	want := "from,to,tag,extra0,extra1\n100,200,Enh,x,y\n"
	if got := out.String(); got != want {
		t.Fatalf("unexpected chr_hmm output:\n%s", got)
	}
}
