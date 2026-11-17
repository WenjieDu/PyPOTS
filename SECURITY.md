# Security Policy

## Reporting a vulnerability

Please report suspected security vulnerabilities **privately** using [GitHub private vulnerability reporting](https://github.com/WenjieDu/PyPOTS/security/advisories/new) (**Security → Report a vulnerability**).

If the private reporting form is unavailable, email the maintainer at [wenjay.du@gmail.com](mailto:wenjay.du@gmail.com) with the subject **PyPOTS security report**.

**Do not** disclose vulnerability details in a public issue, pull request, or discussion before a coordinated disclosure.

When reporting, please include, if possible:

- A description of the vulnerability, its potential impact, and any prerequisites for exploitation.
- The affected PyPOTS version or commit, operating system, Python version, and relevant dependencies.
- Minimal steps to reproduce, a proof of concept, and relevant logs (with secrets and sensitive data removed).
- Any known workaround or suggested mitigation.

Reports in English or Chinese are welcome. We will review reports on a best-effort basis, work with reporters to coordinate remediation and disclosure, and credit researchers with their permission. We cannot guarantee a fixed response or patch timeline.

请通过 [GitHub 私密漏洞报告](https://github.com/WenjieDu/PyPOTS/security/advisories/new)提交安全问题；如无法使用，可通过上述邮箱联系维护者。请不要在公开 Issue 或 PR 中披露漏洞细节。中文报告同样欢迎。

## Supported versions

Security fixes are prioritized for the latest stable release and the actively maintained code. Older releases are generally not backported.

| Version / branch | Support |
| --- | --- |
| Latest stable PyPOTS release | Supported |
| `main` (stable / release branch) | Supported |
| `dev` (active development branch) | Supported |
| Older releases | No guaranteed security fixes; please upgrade |

## In scope

Examples of potentially actionable security issues include:

- Unintended code execution or privilege escalation caused by PyPOTS handling untrusted inputs.
- Unauthorized access, modification, or disclosure of files or sensitive data attributable to PyPOTS.
- Security flaws in PyPOTS packaging, distribution, or its integration with dependencies.
- Other vulnerabilities that cross a reasonable security boundary in PyPOTS itself.

## Out of scope

The following are usually handled as regular bug reports or upstream issues, rather than security vulnerabilities:

- Model accuracy, algorithmic correctness, benchmark results, or ordinary runtime errors without a security impact.
- Vulnerabilities solely in third-party libraries without a PyPOTS-specific impact; report these to the relevant upstream maintainer.
- Malicious behavior of code, plugins, or model files that a user deliberately chooses to trust and execute, unless PyPOTS introduces an additional exploitable security flaw.

PyPOTS is a Python machine-learning library, **not a sandbox**. Do not execute untrusted code or load untrusted serialized models/checkpoints. Some Python and PyTorch serialization formats can execute code when deserialized. Always verify the provenance of model artifacts and dependencies before using them.

## Coordinated disclosure

Please allow maintainers an opportunity to investigate and prepare a fix before publishing technical details. When appropriate, we will coordinate disclosure with the reporter and publish a security advisory or release notes describing the impact, affected versions, and mitigations.
