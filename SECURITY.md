# Security Policy

## Reporting a vulnerability

Please report suspected security vulnerabilities **privately** using [GitHub private vulnerability reporting](https://github.com/WenjieDu/PyPOTS/security/advisories/new) (**Security → Report a vulnerability**).

If the private reporting form is unavailable, email the maintainer at [wdu@time-series.ai](mailto:wdu@time-series.ai) with the subject **PyPOTS security report**.

**Do not** disclose vulnerability details in a public issue, pull request, or discussion before a coordinated disclosure.

When reporting, please include, if possible:

- The affected API or component, what an attacker can control, the prerequisites for exploitation, and the potential security impact.
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
- PyPOTS-specific security flaws in packaging, distribution, CI/CD, or release artifacts that could compromise users.
- Other vulnerabilities that cross a reasonable security boundary in PyPOTS itself.

## Out of scope

The following are usually handled as regular bug reports or upstream issues, rather than security vulnerabilities:

- Model accuracy, algorithmic correctness, benchmark results, ordinary runtime errors, and resource exhaustion from invalid or unusually large local inputs, unless there is a concrete security impact.
- Vulnerabilities solely in third-party libraries without a PyPOTS-specific impact; report these to the relevant upstream maintainer.
- Malicious behavior of third-party scripts, plugins, dependencies, or model files that a user deliberately chooses to trust and execute, unless PyPOTS introduces an additional exploitable security flaw.

For non-security bugs, please open a [regular GitHub issue](https://github.com/WenjieDu/PyPOTS/issues) instead of a private security report.

PyPOTS is a Python machine-learning library, **not a sandbox**. Do not execute untrusted code or load untrusted serialized models/checkpoints. Some Python and PyTorch serialization formats can execute code when deserialized. Always verify the provenance of model artifacts and dependencies before using them.

## Responsible security research

Please test only systems and data you own or are authorized to access. Avoid exposing private information, disrupting services, or using real credentials or sensitive datasets in proof-of-concept reports. Share sensitive reproduction details only through the private reporting channel.

## Coordinated disclosure

Please allow maintainers an opportunity to investigate and prepare a fix before publishing technical details. We will assess reported issues on a best-effort basis, may request clarification or a minimal reproduction, and aim to keep reporters informed as remediation progresses. When appropriate, we will coordinate a disclosure date with the reporter and publish a security advisory or release notes describing the impact, affected versions, and mitigations.
