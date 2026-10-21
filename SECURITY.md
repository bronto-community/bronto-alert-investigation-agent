# Security policy

## Supported versions

Only the latest release is supported. Security fixes land on `main` and ship in
the next release. Each `v*` tag publishes the container image to
`ghcr.io/bronto-community/bronto-alert-investigation-agent` and
`docker.io/brontoio/bronto-alert-investigation-agent`, with provenance and SBOM
attestations on both. If you run a fork or a copy, pull fixes forward from
`main`.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report it privately through
[GitHub's private vulnerability reporting](https://github.com/bronto-community/bronto-alert-investigation-agent/security/advisories/new).
That opens a draft advisory that only the maintainers can see. If you can't use
it, email **security@bronto.io**.

Please include:

- the commit you tested against, and the Python version,
- what an attacker gains, in one sentence,
- the smallest reproduction you have (a failing script is ideal),
- a suggested fix, if you have one.

## What to expect

| Stage | Target |
| --- | --- |
| Acknowledgement of your report | 3 working days |
| Initial assessment, with a severity | 10 working days |
| Fix merged, or a dated plan if it needs longer | 90 days |

We will keep you updated as the assessment goes on. We will credit you in the
advisory unless you'd rather we didn't, and agree the disclosure timing with
you. There is no paid bounty.

## Scope

In scope: anything in this repository. That includes the three entry points
(CLI, Slack bot, webhook receiver), the graph and its phases, and the example
deployment configuration under `deploy/`.

Out of scope:

- Vulnerabilities in the Bronto platform or its APIs rather than in this code.
  Report those to **security@bronto.io** directly.
- Findings in third-party dependencies that have no exploitable path through
  this project. Please report those upstream, though we welcome a heads-up.
- Prompt injection that only changes what the report says. The agent reads
  untrusted log data by design, and its tools are read-only.

## Credential handling

This project handles Bronto API keys, model-provider keys, Slack tokens and a
webhook secret, so reports in this area are particularly welcome. By design, a
credential must never reach a log line, an exception message, a trace span or a
Slack message. Any case where one does counts as a vulnerability, even without
a further exploit. So does a webhook request that is acted on without the
correct secret.
