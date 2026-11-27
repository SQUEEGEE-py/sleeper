# advisor-trigger (Cloudflare Worker)

Starts `.github/workflows/nightly.yml` on time. GitHub's own `schedule:` runs start 5-30
minutes late (once 7 hours) and only in UTC; a `workflow_dispatch` through the API starts within
seconds. Free Workers plan: two Cron Triggers, two requests a day.

One report a day at **7:00 am America/Denver** (user's choice, 2026-11-27; the afternoon
reminder was dropped):

| Cron (UTC) | Does |
|---|---|
| `0 13 * * *` | daily report, only when it is 7 am in Denver (MDT) |
| `0 14 * * *` | daily report, only when it is 7 am in Denver (MST) |

Runs it starts are named `nightly advisor (cloudflare-morning)`.
`nightly.yml` keeps its own schedule 30 minutes later (7:30 am MT) as a backup, which skips
itself when the Worker's run started in the last 3 hours and did not fail. A failed dispatch
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
