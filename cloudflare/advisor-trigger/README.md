# advisor-trigger (Cloudflare Worker)

Starts `.github/workflows/nightly.yml` on time. GitHub's own `schedule:` runs start 5-30
minutes late (once 7 hours) and only in UTC; a `workflow_dispatch` through the API starts within
seconds. Free Workers plan: three Cron Triggers, three requests a day.

| Cron (UTC) | Does |
|---|---|
| `47 12 * * *` | morning report, only when it is 6:47 am in Denver (MDT) |
| `47 13 * * *` | morning report, only when it is 6:47 am in Denver (MST) |
| `43 19 * * *` | afternoon reminder (`only_if_due_today`) |

Runs it starts are named `nightly advisor (cloudflare-morning)` / `(cloudflare-reminder)`.
`nightly.yml` keeps its own schedule 30 minutes later as a backup, which skips itself when the
Worker's run of the same kind started in the last 3 hours and did not fail. A failed dispatch
is posted to Discord.

## Secrets (entered by the user in the Cloudflare dashboard, never committed)

Worker -> Settings -> Variables and Secrets, type **Secret**:

- `GITHUB_TOKEN` -- fine-grained token, repository `SQUEEGEE-py/sleeper` only, permission
  **Actions: Read and write**. Note its expiry date; when it expires the Discord alert says so.
- `DISCORD_WEBHOOK_URL` -- the advisor's webhook.

## Deploying

`wrangler.toml` holds the name, entry point and crons, so `npx wrangler deploy` from this
folder (with a Cloudflare API token from the "Edit Cloudflare Workers" template) reproduces it.
It was first deployed through the Cloudflare REST API from a Claude Code session (2026-11-27).

`worker.js` exports `planFor(cron, date)` so the Denver-clock logic can be checked with node.
