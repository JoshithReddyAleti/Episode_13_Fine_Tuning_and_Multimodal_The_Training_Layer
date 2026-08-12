# Security Policy

## Supported Versions

This is educational content bundled with the AI Engineering Roadmap 2026 newsletter series. The code is illustrative; real production hardening is discussed in Episode 14 (Security).

## Reporting a Vulnerability

For repo-content-related issues (broken links, misleading claims, factual errors):
- Open a GitHub issue with the "security-content" label.

For vulnerabilities in the illustrative code that could affect anyone naively deploying it:
- Please do **not** file a public issue.
- Reach out via the newsletter's contact channels or LinkedIn DM to Joshith Reddy Aleti.

## Common areas to watch when adapting this code

- Never commit API keys or `.env` files. `.env.example` is a template.
- Training data may contain PII. Always run the PII scrub pipeline.
- Multi-tenant LoRA serving requires proper tenant isolation.
- Multimodal input pipelines are attack surfaces (malformed media, PII in images).
- Voice cloning without consent has significant legal and ethical risks.
