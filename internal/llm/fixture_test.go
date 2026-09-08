package llm

import (
	"context"
	"errors"
	"testing"
)

func TestFixtureProviderReturnsResponsesAndRecordsRequests(t *testing.T) {
	provider := NewFixtureProvider([]Response{
		{Message: "first", Usage: Usage{TotalTokens: 3}},
		{Message: "second", Usage: Usage{TotalTokens: 4}},
	})

	res, err := provider.Complete(context.Background(), Request{
		Messages: []Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Complete first: %v", err)
	}
	if res.Message != "first" || res.TokenCount() != 3 {
		t.Fatalf("unexpected first response: %+v", res)
	}

	res, err = provider.Complete(context.Background(), Request{
		Messages: []Message{{Role: "user", Content: "again"}},
	})
	if err != nil {
		t.Fatalf("Complete second: %v", err)
	}
	if res.Message != "second" || res.TokenCount() != 4 {
		t.Fatalf("unexpected second response: %+v", res)
	}

	requests := provider.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected 2 recorded requests, got %d", len(requests))
	}
	if requests[0].Messages[0].Content != "hello" || requests[1].Messages[0].Content != "again" {
		t.Fatalf("unexpected recorded requests: %+v", requests)
	}
}

func TestFixtureProviderHonorsContextCancellation(t *testing.T) {
	provider := NewFixtureProvider([]Response{{Message: "unused"}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := provider.Complete(ctx, Request{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v, want context.Canceled", err)
	}
}

func TestFixtureProviderReturnsConfiguredError(t *testing.T) {
	wantErr := errors.New("fixture failed")
	provider := NewFixtureProviderWithError(wantErr)

	_, err := provider.Complete(context.Background(), Request{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
}
