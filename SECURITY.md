# Security Policy

## Supported versions

Only the latest release on the `main` branch receives security fixes.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing **security@spectrumlabs.tech**. Include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof of concept.
- Any suggested fix, if you have one.

You can expect an initial response within 48 hours and a patch or mitigation
plan within 14 days for confirmed issues.

## Scope

This library provides structured AI completions via a driver registry. Security
issues in scope include but are not limited to:

- API key exposure or leakage through logs, error messages, or response structs.
- Prompt injection vectors introduced by the library itself (not user-supplied
  prompts — that is the caller's responsibility).
- Driver implementations that transmit data to unintended endpoints.
- Insecure defaults in HTTP client configuration (TLS verification, timeouts).

## Out of scope

- Vulnerabilities in upstream provider SDKs or APIs (report those upstream).
- Theoretical weaknesses without a practical exploit path.
- Issues arising from callers embedding untrusted user input directly into prompts.
