# Community maintenance

## Triage

- New bug reports use the existing `potential bug` label until confirmed. Use `bug` for confirmed bugs and remove `potential bug` / `question` when they no longer apply.
- For an unconfirmed report that cannot be investigated without additional information, ask for the missing details and add `question` while retaining `potential bug`. Do not use this combination for reports awaiting maintainer investigation.
- Use `keep` to exempt an issue from stale automation. Assignments exempt questions and unconfirmed bugs awaiting information; milestones exempt all issues.
- Use `completed` only when the work is actually resolved and ready for eventual closure.
- The automation does not mark, close, or delete branches of pull requests. Maintainers review PRs manually.
- If an issue changes category, remove the previous category's stale label as part of triage. Different stale labels prevent old warnings from starting the wrong closing timer.

## Stale Manager

| Eligible issues | Inactivity before warning | Grace period | Close reason |
| --- | ---: | ---: | --- |
| Unassigned, non-milestoned questions without protected labels | 60 days | 14 days | Not planned |
| Unconfirmed reports explicitly awaiting information (`potential bug` + `question`) | 30 days | 14 days | Not planned |
| Explicitly completed, non-milestoned issues without protected workflow labels (including assigned issues) | 14 days | 7 days | Completed |

Confirmed bugs, enhancements, new algorithms, and other protected work are not subject to inactivity closure. Explicitly labeling an issue completed is a separate resolution decision. Security-related issues, `keep`, work awaiting approval, work in progress, and milestoned issues are exempt from all policies. Assignments do not prevent an explicitly completed issue from being closed. A comment or update resets the inactivity timer for eligible issues.

The workflow runs daily at 08:00 UTC and applies the policy directly once the updated workflow is on the default branch. Manual dispatch also applies the policy on the default branch; non-default refs are skipped.

1. Create the repository labels `stale-question`, `stale-reproduction`, and `stale-completed` if they do not already exist. Do not migrate the legacy `stale` label to these labels; each policy must issue a fresh warning.
2. The legacy `stale` label does not count as a warning in the new policies.
3. To run immediately, open Actions → Stale Manager → Run workflow and select the default branch. This is a live run and may label, comment on, or close eligible issues.

The action uses bounded operations per run. Review exceptions and old stale labels manually; no bulk cleanup or label migration is included in this change.
