// Starts the nightly advisor (.github/workflows/nightly.yml) on time.
//
// GitHub's own `schedule:` runs start 5-30 minutes late, once 7 hours, and only in UTC. A
// workflow_dispatch through the API starts within seconds, so this Worker's Cron Triggers call
// it instead. The workflow keeps its own schedule ~30 minutes later as a backup, which skips
// itself when this Worker's run already went through (see nightly.yml's "backup check").
//
// Secrets (Worker Settings -> Variables and Secrets, type Secret, entered by the user):
//   GITHUB_TOKEN         fine-grained token, this repository only, Actions: read and write
//   DISCORD_WEBHOOK_URL  the advisor's webhook; a failed dispatch is posted there
//
// Cron Triggers are UTC. The morning report is 6:47 am America/Denver, which is 12:47 UTC in
// summer time (MDT) and 13:47 UTC in winter (MST): both fire, and only the one that is 6:47 on
// Denver's clock dispatches.

const REPO = "SQUEEGEE-py/sleeper";
const WORKFLOW = "nightly.yml";
const MORNING_CRONS = ["47 12 * * *", "47 13 * * *"];
const REMINDER_CRON = "43 19 * * *";
const MORNING_HOUR_DENVER = 6;

export function denverHour(date) {
  const hour = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Denver", hour: "numeric", hourCycle: "h23",
  }).format(date);
  return Number(hour);
}

// What a cron firing should do: {trigger, onlyIfDueToday} to dispatch, or null to skip.
export function planFor(cron, date) {
  if (MORNING_CRONS.includes(cron)) {
    if (denverHour(date) !== MORNING_HOUR_DENVER) return null;
    return { trigger: "cloudflare-morning", onlyIfDueToday: false };
  }
  if (cron === REMINDER_CRON) return { trigger: "cloudflare-reminder", onlyIfDueToday: true };
  return null;
}

async function dispatch(plan, env) {
  const url = `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;
  const body = JSON.stringify({
    ref: "main",
    inputs: { trigger: plan.trigger, only_if_due_today: String(plan.onlyIfDueToday) },
  });
  let last = "";
  for (let attempt = 1; attempt <= 3; attempt++) {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "advisor-trigger",
        "Content-Type": "application/json",
      },
      body,
    });
    if (resp.ok) return null;
    last = `HTTP ${resp.status}: ${(await resp.text()).slice(0, 300)}`;
    // Only server errors and rate limits are worth retrying; a 401/403/404/422 will not change.
    if (resp.status < 500 && resp.status !== 429) break;
    await new Promise((r) => setTimeout(r, 2000 * attempt));
  }
  return last;
}

async function alert(env, text) {
  if (!env.DISCORD_WEBHOOK_URL) return;
  await fetch(env.DISCORD_WEBHOOK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: text.slice(0, 1900) }),
  });
}

async function handle(cron, env, now) {
  const plan = planFor(cron, now);
  if (!plan) {
    console.log(`cron ${cron}: nothing to do at Denver hour ${denverHour(now)}`);
    return;
  }
  const error = env.GITHUB_TOKEN ? await dispatch(plan, env) : "GITHUB_TOKEN secret is not set";
  if (error) {
    console.log(`cron ${cron}: dispatch failed: ${error}`);
    await alert(env, `Advisor trigger (Cloudflare) failed to start the ${plan.trigger} run: ${error}\n`
      + "GitHub's backup schedule should still run about 30 minutes later.");
  } else {
    console.log(`cron ${cron}: dispatched ${plan.trigger}`);
  }
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(handle(event.cron, env, new Date(event.scheduledTime)));
  },
  async fetch() {
    return new Response("advisor-trigger: runs on Cron Triggers only\n", { status: 404 });
  },
};
