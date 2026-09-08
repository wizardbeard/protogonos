package llm

import (
	"context"
	"errors"
	"sync"
)

var ErrNoFixtureResponses = errors.New("no fixture responses configured")

type FixtureProvider struct {
	mu         sync.Mutex
	responses  []Response
	requests   []Request
	err        error
	repeatLast bool
}

func NewFixtureProvider(responses []Response) *FixtureProvider {
	return &FixtureProvider{
		responses:  cloneResponses(responses),
		repeatLast: true,
	}
}

func NewFixtureProviderWithError(err error) *FixtureProvider {
	return &FixtureProvider{err: err}
}

func (p *FixtureProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if err := ctx.Err(); err != nil {
		return Response{}, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	p.requests = append(p.requests, cloneRequest(req))
	if p.err != nil {
		return Response{}, p.err
	}
	if len(p.responses) == 0 {
		return Response{}, ErrNoFixtureResponses
	}
	if len(p.responses) == 1 {
		return cloneResponse(p.responses[0]), nil
	}
	res := p.responses[0]
	if p.repeatLast {
		p.responses = p.responses[1:]
	}
	return cloneResponse(res), nil
}

func (p *FixtureProvider) Requests() []Request {
	p.mu.Lock()
	defer p.mu.Unlock()

	out := make([]Request, len(p.requests))
	for i := range p.requests {
		out[i] = cloneRequest(p.requests[i])
	}
	return out
}

func cloneResponses(in []Response) []Response {
	if len(in) == 0 {
		return nil
	}
	out := make([]Response, len(in))
	for i := range in {
		out[i] = cloneResponse(in[i])
	}
	return out
}
