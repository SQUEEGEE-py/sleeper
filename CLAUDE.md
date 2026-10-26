# CLAUDE.md

Context for Claude Code working in this repo. Read `docs/SPEC.md` for the full design and
`docs/TASKS.md` for the milestone list. **Current state and next steps are in the "Current
state" section below** — read it before starting work, it says what is actually built.

## What this is

A decision-support tool for a single 8-team Sleeper NBA fantasy league that uses **Lock-In**
scoring. Two products share one valuation engine:

1. **Draft assistant** — a ranked board plus a live snake-draft advisor.
2. **In-season coach** — a nightly lock-or-roll advisor, waiver and trade suggestions.

The tool **never writes to Sleeper**. Sleeper's public API is read-only. Every recommendation
ends with the user taking the action in the Sleeper app.

## Non-negotiable rules

1. **`config/league.json` is the only source of truth for league rules.** Never hardcode
   scoring weights, roster slots, or team counts anywhere else. On startup, fetch the league
   from Sleeper and assert the config matches; fail loudly on mismatch.
2. **The math is deterministic code, not an LLM.** Projections, distributions, optimal stopping,
   draft simulation, and win probability are all computed in Python and unit tested. The LLM
   layer only explains results, ingests unstructured news, and drafts trade messages. If you
   find yourself asking a model to compute a number, that is a bug.
3. **Lock-In is not standard fantasy basketball.** Only one game per player per week counts.
   Any ranking, projection, or piece of advice copied from standard fantasy logic is wrong here.
   See `docs/SPEC.md` §3.
4. **Never invent data.** If an API is unreachable or a player is unmapped, surface it as an
   explicit gap in the UI. A recommendation built on a silently-defaulted number is worse than
   no recommendation.
5. **Secrets stay in `.env`** (gitignored). No keys, league IDs excepted, in committed code.
6. **Verify third-party endpoints before building against them.** The URLs in the spec were
   accurate when written but these are mostly undocumented or fast-moving APIs. Probe first,
   then write the client.

## Current state (2026-10-22)

**The draft is done** (Sunday 2026-10-18, see "The draft" below). **Phase 3 (in-season) has
started**, following a seven-step plan the user set on 2026-10-19 (see "Phase 3 plan and
status" at the end of this section). **Step 1 is done**: a week-1 lock cheat sheet
(`lock_sheet.py`). **Step 2 is wired in (age-band ratio only, the user's choice)**: a
back-to-back second night lowers that game's `p_play` by the measured ratio for the player's
age on the game date, which lowers the lock thresholds before it. The per-player blend is a
later, separate step. **Step 3 is done** (2026-10-21): every job prints UTF-8 and the live
advisor's whole poll is guarded (`engine/jobs/runtime.py`). The season opened Tuesday
2026-10-20. On 2026-10-21 the user also corrected the league rules (2 IR slots,
reverse-standings priority waivers, no FAAB) and chose a **cloud-first nightly job with
Discord alerts** for step 4, which is built and running (one check left, ~2026-10-27).
**Step 5 is done** (2026-10-22): both scoring flags are true, settled from last season's
league. **Step 6 is done** (2026-10-22): age curve and role watch. **Step 7 is done** (2026-10-22): IR alerts, pickups, trade finder, deadline countdown. **All seven phase 3 steps are done**; what remains is the end-of-plan cleanup (README, docs/SPEC.md) and the follow-ups listed under step 7.

**Working agreement for phase 3:** do the steps in order and **stop after each one** for the
user to review. Commits are `phase3: <what>`. The user asks before anything is committed or
pushed.

Everything runnable today, in the order you would use it:

```
uv run pytest                                             # 419 tests, none need the network
uv run python -m engine.jobs.verify_league                # config vs live Sleeper; exits 1 on drift
uv run python -m engine.jobs.ingest_history [--force]     # LOCAL ONLY: logs + schedule -> parquet
uv run python -m engine.jobs.build_board [--top 150]      # -> data/board/{board.csv,board.html,pool.csv}
uv run python -m engine.jobs.player_value "Jokic" [--compare "Kawhi" ...]
uv run python -m engine.jobs.draft_rehearsal [--slot 5] [--drafts 2]   # synthetic drafts, timed
uv run python -m engine.jobs.draft_rehearsal --calibrate                # 2025 replay -> sigma
uv run python -m engine.jobs.draft_live [--draft-id <mock id>] [--slot N] # -> data/draft/live.html
uv run python -m engine.jobs.lock_sheet [--week N]                      # -> data/lock/lock_sheet.html
uv run python -m engine.jobs.nightly [--week N] [--post]                # win prob + LOCK/ROLL; --post -> Discord
uv run python -m engine.jobs.publish_snapshot                           # data/ -> snapshot/ for Actions; commit it
uv run python -m engine.jobs.probe_endpoints                            # stdlib only; also runs in Actions
uv run python -m engine.jobs.resolve_scoring                            # flags vs Sleeper counted scores
```

Per-file test counts: scoring 89, distributions 38, lockin 37, nba_stats 26, lineup 25,
player_ids 23, draft_sim 29, lock_sheet 17, rookies 15, valuation 11, draft_live 11,
verify_league 8, runtime 4, matchup 14, locks 10, nightly 10, alerts 6, resolve_scoring 5, aging 7, roles 6, roster_moves 10, nightly_roster 6, trades 8, http_retry 5.

`data/` is gitignored and regenerable. **On a fresh checkout run `ingest_history` then
`build_board` before anything else** -- the 2026-10-16 session started with no `data/` at all.
Rebuild the board on draft day to pick up Sleeper injury statuses.

What exists and works:

- `uv` project on Python 3.11 with the M0 deps. Run anything with `uv run python -m <module>`.
- Repo skeleton from SPEC §1. The only **empty placeholder files, not stubs**, left are
  `engine/ingest/markets.py` and `engine/ingest/news.py` (0 bytes on purpose). Real so far: `scoring.py` (M1),
  `nba_stats.py` (M2), `distributions.py` and `lockin.py` (M3), `player_ids.py`, `lineup.py`,
  `valuation.py` and `build_board.py` (M4), `rookies.py` (pre-draft fixes), `draft_sim.py`,
  `draft_live.py` and `draft_rehearsal.py` (M5, cut down), `player_model.py` and
  `lock_sheet.py` (phase 3 step 1), `runtime.py` (step 3), `matchup.py`, `locks.py`,
  `nightly.py` and `probe_endpoints.py` (step 4, in progress).
- `engine/ingest/sleeper.py` — typed pydantic client for the SPEC §2.1 endpoints. Every
  response is cached to disk under `data/cache/sleeper/` with a fetch timestamp; `players/nba`
  is TTL-capped at 24h. Reruns and tests never need the network.
- `engine/jobs/verify_league.py` — diffs live `scoring_settings` and `roster_positions`
  against `config/league.json`, prints a readable diff, exits non-zero on mismatch.

Endpoints probed against the live API on 2026-09-17 per rule 6, using a public league:
`/league/{id}` (`scoring_settings` dict, `roster_positions` list), `/rosters`, `/users`,
`/drafts`, and `/players/nba` (dict keyed by player_id, ~2100 players, 2.4 MB). All match the
shapes the spec describes.

### M0 acceptance test: PASSING as of 2026-09-23

`uv run python -m engine.jobs.verify_league` exits 0 against the real league —
**National Logan League**, `league_id` `1405604996254339072`, 8 teams, 13 roster slots,
scoring confirmed. The first run exited 1 on six mismatches; all six are now resolved and the
resolutions are recorded in `config/league.json`'s `_*_comment` fields:

- **Five scoring keys were named wrong** in the original hand transcription. Sleeper's real
  names are `tpm`, `tf`, `ff`, `bonus_pt_40p`, `bonus_pt_50p`. **Every weight was already
  correct** — only the spellings differed, and the live payload has exactly 13 keys with no
  extras. `config/league.json` now stores Sleeper's spelling verbatim so `verify_league`
  compares with no translation; `scoring.py` holds the one verified
  `STAT_FIELD_TO_SCORING_KEY` table and the rest of the engine speaks readable field names.
- ~~The league has no IR slots~~ -- **wrong, corrected 2026-10-21: there are 2.** Sleeper
  keeps IR in `settings.reserve_slots`, never in `roster_positions` or the draft's `slots_*`,
  which is all this check looked at. See "What the live league changed".

### The league itself

**National Logan League**, `league_id` `1405604996254339072`, draft `1405604996266901504`
(snake, 13 rounds, 90s pick timer, **`complete`**). The user is **`Squ33gee`**, Sleeper
`user_id` `1152443723016556544`, **roster_id 4, draft slot 1** -- roster_id and draft slot
are different numbers (`slot_to_roster_id` maps them); key rosters on roster_id. The other
seven managers: `tristantro`, `amancito`, `2jake4davis6`, `aala1aala2`, `cyapp1`, `gilbobo`,
`ChickenJoe2107`.

`previous_league_id` `1275683545209122816` is last season's league with the same managers.
Its completed draft (`1275683545217503232`) **is fetched and used**: it calibrated the draft
advisor's survival model (see M5) and shows per-coach tendencies (cyapp1 reaches ~25%).

### Blockers, in priority order

None open as of 2026-10-22. The nightly advisor runs in GitHub Actions and posts to Discord
(step 4), and both scoring flags are resolved (step 5, below). What is left is the phase 3
plan itself.

### What the live league changed vs docs/SPEC.md

The spec was written from screenshots. `config/league.json` has been corrected; **the spec
has not been rewritten yet.** Two of the 2026-09-23 "corrections" were themselves wrong and
were fixed on 2026-10-21 -- the lesson for both: **league rules live in `settings`, not only
in `roster_positions`, and a field's presence is not evidence the feature is on.**
`verify_league` now checks `settings` too (IR slots and eligible statuses, waiver type, clear
days, game-start waivers, playoffs, trade deadline, divisions, teams; `test_verify_league.py`)
and fetches the league live instead of from the forever-cache.

- **2 IR slots -- the spec was right.** `settings.reserve_slots` = 2; one roster already uses
  it. Eligible: **OUT, SUS, DNR** only (`reserve_allow_*`), not DTD or doubtful. An IR player
  costs no active spot, so SPEC §7.2's IR stash applies -- and matters most to the user, whose
  roster is the oldest in the league.
- **No FAAB -- priority waivers.** Confirmed by the user in the app. `waiver_budget: 100` is a
  default present in every league; week 1's waiver claim has `{seq: 0}` and no `waiver_bid`,
  every `waiver_budget_used` is 0, and each roster has a `waiver_position` (the user was 7th
  of 8 on 2026-10-21). **Order is reverse standings** (user-confirmed), so `waiver_type` 1 =
  reverse standings: a winning claim does not send you to the back, and leading the league
  means claiming last. Dropped players clear in 1 day; a free agent goes back on waivers
  once his game tips (`waiver_after_game_start`), so adds must happen before his game. There
  is no bid-sizing problem; the question is when a claim is worth spending priority on.
- **Two divisions**, which the spec never mentions. Check how they affect the top-4 playoff
  cut that the draft simulator's `threat[team]` is built around.
- **No keepers, despite `max_keepers: 1`.** That field is an inert Sleeper default here;
  `settings.type` is 0 (redraft) and all 8 pre-draft rosters are empty. Every player is
  draftable. Recorded in `config/league.json` so it does not get reopened from `max_keepers`
  alone.
- Non-issue, but do not be alarmed by it: the league-level `settings.draft_rounds` is `3`. It
  is a stale default. The draft object's `rounds: 13` is the real value, giving the 104 picks
  the spec assumes.

### M1 — scoring engine (done)

`engine/model/scoring.py` implements SPEC §4. `uv run pytest tests/test_scoring.py` is green:
89 tests, 10 hand-scored fixtures × both settings of both ambiguity flags, no network.

How it is put together, so it does not get re-litigated:

- Weights are a **signed dot product** over `config["scoring"]`. The config already carries the
  minus signs on turnovers and fouls, so the module applies no sign of its own and hardcodes no
  number. The only hardcoded things are basketball: the five countable categories, and that a
  "double" is 10.
- Points-bonus thresholds are **parsed from the config key names** (`bonus_pt_(\d+)p`), so a
  commissioner adding `bonus_pt_60p` needs no code change. There is a test for that.
- `load_rules` fails loudly on a missing weight, an **unrecognised** scoring key, or an unknown
  `quadruple_double` policy. The last is validated at load time specifically so a quad-double
  can never crash a live nightly job.
- `score_breakdown()` returns the itemised parts for the UI's one-line "why"; `score_box()` is
  the total. Everything downstream calls one of those two.
- The expected values in `tests/fixtures/box_scores.json` were hand-computed and user-reviewed
  before the test file was written. **They are the reference — if scoring.py disagrees,
  scoring.py is wrong. Never regenerate them from code output.** No test recomputes a total
  from the weights, since a test that reimplements the formula only proves it equals itself.
- Verified by mutation: flipping non-stacking to pay the lowest threshold, and disabling the dd
  bonus, each fail 5 and 8 tests respectively.

### M2 — historical ingest (done)

`engine/ingest/nba_stats.py` + `engine/jobs/ingest_history.py`. Run it by hand:
`uv run python -m engine.jobs.ingest_history [--force]`. 26 tests, no network.

On disk after a run (all gitignored and regenerable): **79,358 game-log rows** across 2023-24,
2024-25 and 2025-26, plus the 2026-27 schedule, week grid, and the team x week games grid that
M2's acceptance test asks for.

What was learned building it, so it is not rediscovered:

- **cdn.nba.com is blocked from this machine** (Akamai 403, with and without browser headers),
  so SPEC §2.2's lighter-weight schedule source is unavailable. stats.nba.com works fine and
  supplies both the logs and, via `ScheduleLeagueV2`, the schedule and week grid.
- **The NBA's own week numbering IS this league's fantasy week numbering** — week 1 starts
  2026-10-20 matching config `season_start`, every later week runs Monday to Sunday, 25 weeks
  covering the week-22 fantasy playoffs. This is load-bearing for M3, so it is *verified at
  ingest time* by `verify_weeks_match_fantasy_weeks`, which fails the job rather than
  assuming. Do not replace it with a hand-rolled week calculation.
- **Two real data quirks, both fixture-tested.** `gameDate` is a US-format string, so sorting
  it as text puts January before December and corrupts every week assignment. And the week
  frame arrives unsorted with week 8 listed twice, identically — identical rows are collapsed,
  but rows that disagree about a week's dates are deliberately left to fail the verifier.
- **Two games per team are unscheduled** (6 placeholder rows with both sides TBD, plus 24
  games not yet listed; every team has 80 of 82). NBA Cup. `schedule_completeness()` reports
  it and the job prints it. Games-per-week in that window is a floor, not a count — which
  matters because games per week feeds lock-in value directly.
- **Technicals and flagrants are not in `PlayerGameLogs`.** The league scores them; the
  endpoint only has personal fouls. Stored as 0.0 and listed in `GAME_LOG_MISSING_STATS` so
  the UI can mark them unmeasured. Recovering them needs play-by-play.
- **The current season is fetched too, and always refetched** (`ingest_current_season`).
  Completed seasons are cached because they never change; the season being played is not,
  because changing is the point. Before opening night it returns an empty frame that still
  carries the right columns, so no caller needs a September special case. Preseason goes to a
  separate `preseason_logs_{season}.parquet`. The endpoint spells it **"Pre Season"** with a
  space — get it wrong and you get an empty frame indistinguishable from "season not started",
  so a test pins the constant.

Sanity check, rescoring real 2025-26 logs through `scoring.py`: the top of the board is
Dončić 68.9, Jokić 67.8, SGA 59.1, Wembanyama 57.0 mean fp/game, league mean 23.3 — exactly
the shape this scoring system should produce. And **Jokić is the player most affected by the
two unresolved ambiguity flags** (+1.66 fp/game between the extremes), which is precisely what
SPEC §4 predicted.

### M3 — distributions and lock-in value (done)

`engine/model/distributions.py` + `engine/model/lockin.py`, driven by
`uv run python -m engine.jobs.player_value "<name>" [--compare ...]`. 56 tests, no network.

**The closed-form regression passes exactly**: normal(45, 10) gives 45.000 / 48.989 / 51.297 /
52.904 for 1-4 game weeks against the 45.0 / 49.0 / 51.3 / 52.9 target, and a 200k-draw
simulation through the empirical path reproduces it to within 0.15. Both paths are tested,
because the analytic one alone would only prove the normal formula matches itself.

Design notes worth keeping:

- **`lockin.py` is pure and distribution-agnostic.** It takes anything satisfying the
  `ScoreDistribution` protocol (`p_play`, `mean()`, `expected_max(c)`). Two implementations:
  an analytic normal for the regression and for reasoning, and a weighted empirical sample for
  real players. The conditional distribution is deliberately **not** fitted to a normal -- the
  40+/50+ bonuses live in the right tail and a normal would smooth them away.
- **The terminal game is E[X_n], not E[max(X_n, 0)].** There is no decision left at the last
  game, and scores can be negative (no points, three turnovers). Running the general recursion
  at the terminal step would quietly assume the auto-lock fallback protects you from a
  negative. It does not. There is a mutation-tested case for this.
- **`p_play` is measured per team stint**, so a traded player is not charged for both teams'
  full seasons. The honest limit, documented in the docstring: each window starts at his first
  appearance, so games missed at the very start or end of a stint are invisible. It is a base
  rate for in-stint availability; injury designations adjust it at decision time.
- **Recency weighting is by rank, not elapsed time** (half-life 25 games). A player who missed
  three weeks injured has not become three weeks staler than his last game suggests.
- **Shrinkage toward a minutes-band role prior** is what stops a six-game sample producing a
  confident number. `history_reliable` flags anyone under 20 games for SPEC §5.3's
  unreliable-history column.

Verified by mutation, like M1: using the general recursion at the terminal step, dropping
`p_play` from the roll-on branch, and an off-by-one in the thresholds each fail the suite (1,
4 and 12 tests respectively).

### M3 spot checks, on real data

Both of `docs/TASKS.md` M3's eyeball tests hold:

- **Volatility beats a higher mean.** Anthony Edwards (mean 52.5, sd 17.4) outranks Jaylen
  Brown (mean 52.8, sd 13.9) on season value despite the lower mean. Wembanyama (58.6/17.6)
  outranks SGA (58.5/13.6) despite worse availability. At a fixed mean of 45 in a 3-game week,
  sd 8 -> 50.04 and sd 20 -> 57.59.
- **Missed games are cheap.** Kawhi Leonard misses 16.3% of games and loses only **6.5%** of
  season value; Jokic misses 13.0% and loses 5.3%; Gobert misses 7.8% and loses 3.3%. Across
  the board the value loss runs at roughly **40% of the games-missed rate** -- which is the
  whole reason rest-prone stars are underpriced by drafters using standard rankings.

### M4 — the draft board (done) 🎯

`uv run python -m engine.jobs.build_board [--top 120]` writes `data/board/board.csv` and
`data/board/board.html` (gitignored, regenerable). 58 tests.

**The HTML page is fully self-contained** — 39 KB, zero external references, no fetch, no CDN,
no web fonts, dark mode, sticky header, client-side filter and sort. Verified by scanning the
output for any `src`/`href` leaving the file. That is its whole job: to still work on a phone
in the draft room when the wifi or the live advisor has died.

Pieces: `engine/ingest/player_ids.py` (identity), `engine/model/lineup.py` (slot assignment),
`engine/model/valuation.py` (replacement level and VOR), `engine/jobs/build_board.py`.

- **Regression found and fixed 2026-10-22:** Sleeper's `search_full_name` started carrying
  generational suffixes run together ("jabarismithjr"), where the suffix strip cannot see a
  word boundary, and the matcher preferred that field -- so all 32 Jr./II/III players went
  unmapped overnight and dropped out of `pool.csv`. Each player is now indexed under both
  `full_name` and `search_full_name`; 808/809 map (1 real duplicate name). The gap list is
  what caught it, which is exactly why gaps are printed and never dropped.
- **Player identity: 506/506 mapped, no gaps.** Sleeper's NBA dictionary has no NBA.com id
  (it has sportradar, swish, rotowire, kalshi — not this one), so the join is by name.
  Normalising accents, punctuation and generational suffixes, then breaking ties by team, maps
  everyone. Unmapped and ambiguous players are returned and printed and shown on the board,
  never dropped. Manual fixes go in `config/player_overrides.csv`.
- **`lineup.py` is exact, not greedy** — `scipy.optimize.linear_sum_assignment`. There is a
  test for the case greedy gets wrong: a flexible player taking the only C slot and stranding
  a better pure centre on the bench.
- **Shrinkage toward a minutes-band role prior is wired in, and it mattered.** Without it the
  board filled with volatile low-minute rookies, because Lock-In rewards variance and a thin
  sample is almost all variance. With it, thin-history players in the top 120 dropped to 2,
  none in the top 50, and mean minutes of the top 120 is 30.9.

#### The replacement-level trap (do not undo this)

Everyone is UTIL-eligible and the greedy fills UTIL last, so **the marginal UTIL starter is
structurally the weakest starter in the league**. Taking the minimum replacement across all of
a player's eligible slots therefore hands every single player the UTIL number, and VOR
collapses to "season value minus a constant" — the flat player-number-72 cutoff SPEC §5.3
explicitly forbids. The first build did exactly this: all 120 rows shared one replacement
value.

`replacement_for` now excludes UTIL from that minimum, so a player is measured against his
dedicated positional slots. `test_vor_is_not_a_flat_cutoff` guards it. The three resulting
bars are the marginal starter in each position family: **centres 40.1** (16 jobs),
**forwards 41.4** (24 jobs), **guards 42.5** (24 jobs).

#### M4's eyeball tests: two pass, one is wrong in the spec

Measured by re-scoring 2025-26 with individual scoring rules switched off, which isolates each
mechanism instead of comparing against an arbitrary baseline:

- ✅ **Defensive specialists rank lower.** Clearest effect in the whole board. Top-quintile
  steals+blocks share shifts −14.9 rank places against a conventional points league; bottom
  quintile +6.2. Steals and blocks are worth 1 here against 2 almost everywhere else.
- ✅ **The double-double bonus does lift double-double bigs.** Turning `dd`/`td` off moves
  centres −1.4 and 40%+ double-double centres −2.9; correlation between double-double rate and
  the shift is **+0.476**. The mechanism works exactly as SPEC §6.3 describes.
- ❌ **But centres still do not out-rank standard fantasy, because the 1.5x points multiplier
  is much larger than the double-double bonus.** Turning the multiplier and the 40+/50+
  bonuses off moves centres **−4.9** and 40%+ double-double centres **−8.1**, against +2.1 for
  non-centres; correlation with points per game **+0.176**. The +3 bonus is worth about +1.5 a
  game to a big who doubles half the time, while 1.5x turns a 25-point scorer's 25 into 37.5
  and a 10-point centre's into 15 — a far bigger differential. Net across the full 506-player
  universe, centres sit **−3.6** rank places against a conventional baseline, not above it.

  **Consequences.** `docs/TASKS.md` M4's eyeball expectation ("centres with reliable
  double-doubles rank above where standard fantasy rankings put them") is half right: the
  bonus helps them, the scoring system as a whole does not. And SPEC §6.3 names "reliable
  double-double bigs" as one of two places where denial is genuinely large — that exception
  rests on a premium the data does not support, so it should be re-examined before M5 builds
  blocking policy on it. Centre *scarcity* is real and is captured by the replacement level
  (40.1, the lowest bar); centre *production* is not specially rewarded here.

### Still owed from earlier milestones

What to do next is in "Phase 3 plan and status" at the end of this section.

- **Supabase is not set up.** No `.env`, so `ingest_history` writes parquet only and says so.
  TASKS M2 asks for parquet *and* Supabase; that half is deferred, not done.
- **`config/player_overrides.csv` is wired in but empty — and mostly it should stay that
  way.** The user's call, and it is the right one: once the season starts the model learns new
  roles by itself, so hand-entered multipliers are only worth it for the pre-season window.

  `ingest_current_season` now pulls the season being played on every run (always refetched,
  never cached) and the board folds it in, so this happens with no further work. Recency
  weighting at a 25-game half-life gives the current season **24% of a player's weight after
  10 games (~3 weeks), 50% after 25 (~8 weeks), 75% after 50 (~16 weeks)**.

  What that does *not* cover is **draft night itself**, when zero current-season games exist.
  Preseason (`preseason_logs_{season}.parquet`) is now read by the board, but only for
  *minutes* (rookies and thin histories, below), never as distribution data.

### Pre-draft board fixes (2026-10-16)

Auditing the board against Sleeper's own ranking (`search_rank`) found it silently missing
players the other managers will draft. All fixed; `build_board` now also runs a **safety net**
that lists as a gap any Sleeper top-200 player who is not valued. It is empty today.

- **Stars who missed last season were dropped.** The candidate filter required 10 games in
  2025-26, which removed Haliburton (Sleeper #17), Irving, Lillard and VanVleet. Now 10 games
  in *any* season of the window qualifies; those players carry a `missed 25-26` flag. A
  player Sleeper lists with no team is left off (retired or unsigned) -- rotation players
  among them (20+ min) are listed as a gap instead. The team used is always Sleeper's current
  one; nothing falls back to last season's team any more.
- **Rookies were absent.** `engine/model/rookies.py` values them. Measured, so not re-litigated:
  - Preseason minutes predict regular-season minutes weakly: `regular = 0.90 x pre + 7.0`,
    r = 0.47, residual sd 6.8 min. So each rookie shows a VOR **range** at +/- 1 sd.
  - Rookies score **0.908x** a veteran at equal minutes (whole-minute bins), with the spread
    lower by about the same factor, so the distribution is scaled, not shifted.
  - The cohort is *all* player-seasons near the projected minutes, all of their games,
    equal weight per player-season. A rookies-only cohort was tried first and failed: only
    three 2024-26 rookies averaged 30+ minutes, so value stopped rising with minutes.
  - **Known weakness: no draft pedigree.** The #1 pick and a second-rounder at equal
    projected minutes value the same, so top picks (Dybantsa, Peterson, Boozer) come out low.
    Flagg's rookie season (43.8 fp at 33.5 min) beat veterans at those minutes. The board
    shows Sleeper's rank beside ours; the high end of the range is the elite-pick case.
- **Late-season tank call-ups were overrated.** Kennedy Chandler (11 games at 32 min for Utah,
  Mar-Apr 2026) ranked 59th because his shrinkage prior's band was picked from the same
  inflated minutes. For anyone under 20 games the band now comes from preseason-projected
  minutes.
- **Board additions:** `Slpr` column (Sleeper's rank = what opponents see by default; the gap
  to our rank is who can be waited on), Sleeper injury status as a flag, default 150 rows.

**Rejected: a preseason role-change report for veterans.** Preseason minutes cannot see role
changes: starters are rested (Jokic 17.9 min, Maxey 18.5, Durant 19.7), so every star reads as
a 7-15 minute drop. The meaningful veteran signal is a team change, which the board already
handles for schedule. 19 of the top 110 changed teams (Giannis MIA, Brown and LeBron PHI,
Kawhi TOR, AD WAS, ...); their *role* is still last season's, and any adjustment is a manual
`config/player_overrides.csv` row.

**Judgment calls left to the user, deliberately:** Lillard is our #23 vs Sleeper #146 (torn
Achilles, age 36; the model only knows 2024-25). Rookie values are wide ranges.

### M5, cut down — live draft advisor (built 2026-10-16)

The full M5 (per-coach updating, positional-need opponents, threat, blocking slider) did not
fit before the draft. What was built instead, agreed with the user, who drafts **on the
laptop** with the static board on the phone as the fallback:

- `engine/model/draft_sim.py` (pure, 29 tests): snake arithmetic, survival simulation,
  roster-aware value, one-step lookahead. Its docstring has the full reasoning.
- `engine/jobs/draft_live.py` (10 tests): polls the draft every 3 s, writes
  `data/draft/live.html` (self-contained, reloads itself) and prints the same to the
  terminal. Identifies the user by Sleeper username (`/user/{username}`), so it works on a
  mock draft by `--draft-id` too; slot comes from `draft_order`, else his own picks, else
  `--slot`. Before the slot is known it still ranks by value.
- `engine/jobs/draft_rehearsal.py`: synthetic full drafts through the same `build_view`, and
  the 2025 calibration replay.

Decisions and measurements, so they are not re-litigated:

- **Opponents follow Sleeper ADP**, from the undocumented
  `api.sleeper.com/projections/nba/{season}` (`stats.adp_std`), probed 2026-10-16; 2026 rows
  update daily. On this league's 2025 draft, 2025 ADP vs actual pick: **Spearman 0.948**.
  Caveat: the 2025 rows were last modified Jun-Jul 2026, so they may not be the exact
  draft-day snapshot.
- **Noise sigma = 0.30, from calibration, not from the pick spread.** `sd(log(pick/ADP))` is
  0.21, but 0.21 is overconfident when predicting who lasts (3% predicted, 11% actual). 0.30
  is calibrated in every bin and has the best Brier (0.0978) over 0.15-0.40.
  `draft_rehearsal --calibrate` reproduces the table.
- **Roster value is lineup gain in VOR units**, with an empty slot worth 0. Do **not** switch to
  replacement-level phantom players: every phantom chains into UTIL and value collapses to a
  flat cutoff, the same trap as M4's (`test_gain_is_not_a_flat_cutoff`).
- **Lookahead:** score = gain now + expected best gain still there at the following pick,
  over 2000 sims, given he is taken. This is what makes it take the player about to go over a
  slightly better one who will last (`test_takes_the_player_who_will_be_gone_...`).
- **Failure behaviour:** a fetch error keeps the last good state and shows a red banner with
  its age; the page turns red on its own (inline JS) if not rewritten for 20 s, meaning the
  process died. Pages are written atomically. Verified with the network blocked.

Rehearsal results: 16 synthetic drafts (2 per slot), **slowest pick 0.57 s** against the
10 s budget, every lineup filled.

**Real Sleeper mock draft, 2026-10-16** (`1417017638327988224`, a `league_mock` with the
league's exact settings; user drew slot 8). All 104 picks in ~7 minutes against Sleeper CPUs,
no fetch errors, recompute 0.3 s, slot picked up from `draft_order` the moment it started.
The user followed the top recommendation all 13 times; every starting slot filled. One bug
found and fixed: at the back-to-back turn (#8/#9) float noise listed Brown above Tatum
although the pair scores identically -- scores are now rounded before ranking. Starter VOR
was 103.7 vs 34-67 for the CPUs, but that is measured by our own valuation against teams
drafting by Sleeper's, so it shows the plumbing works, not that the model is right.

**Known limits, all deliberate for draft night:**

- Bench picks have zero lineup gain, so once the nine starters are set the advice is plain
  best-VOR-available. In Lock-In, bench depth is injury cover, which this does not value.
- No per-coach model. cyapp1 took players ~25% ahead of ADP all through 2025, the largest
  coach effect, and the first thing to add.
- No positional need in the opponent model, so it may slightly overrate the survival of a
  position everyone still needs late.
- `enforce_position_limits` is on in the draft settings; the limits themselves are unknown.

### The draft (2026-10-18)

**Preparation the same day.** Re-ran `ingest_history` and `build_board`: NBA final roster cuts
had landed since Friday, so Sleeper's on-a-team count was 565 and the valued pool fell from 565
to 506 -- real movement, not a bug; the top-200 safety net stayed empty. Checked and clean:
`verify_league` exit 0; no traded picks (`/draft/{id}/traded_picks` and
`/league/{id}/traded_picks` both `[]`, `draft_pick_trading` false). A scripted mid-draft outage
(real `draft_live.main()` against a fake Sleeper: ok, 4 failed polls, ok) kept the last state
with the red banner, then recovered and jumped 20 picks correctly. **Found: the laptop sleeps
after 3 minutes idle on battery** (`powercfg` DC standby 180 s; AC is 3 h) -- the user plugged
in. Not changed on purpose that close to the draft: the rookie blend and an age adjustment.

**Second mock** (`1417673165013852160`, user drafting by hand on the real clock, advisor run
from his own terminal): advisor tracked all 104 picks, the user took its #1 recommendation 13
of 13 times. It produced almost the same roster as the first mock -- old, and with four guard
bench players and no backup centre, because bench gain is zero. The user was told to take a
centre late when one was in the top three; he did in the real draft (Vucevic).

**The real draft.** Slot 1 (order set at ~18:00, the advisor picked it up from `draft_order`).
Final roster: starters Jokic, Jaylen Brown, Durant, Bam Adebayo, Kawhi, Lillard, LeBron,
Kyrie, Keyonte George; bench DeRozan, Saddiq Bey, Vucevic, Caleb Wilson. Every slot filled,
three C-eligible.

How the eight teams compare (lineup VOR is our model; median ADP is Sleeper's market view,
lower = more highly rated):

| Manager | Slot | Lineup VOR | Weekly lock-in pts | Median ADP | Avg age | 33+ |
|---|---|---|---|---|---|---|
| **Squ33gee** | 1 | **107.8** | **486.2** | 79.6 | **31.8** | **7** |
| gilbobo | 3 | 70.3 | 446.7 | 33.3 | 26.5 | 1 |
| ChickenJoe2107 | 4 | 60.4 | 438.3 | 52.0 | 26.5 | 0 |
| cyapp1 | 8 | 60.3 | 434.1 | 76.2 | 27.8 | 3 |
| aala1aala2 | 6 | 49.7 | 428.5 | 51.6 | 27.3 | 0 |
| amancito | 7 | 45.3 | 413.3 | 45.3 | 27.0 | 1 |
| 2jake4davis6 | 5 | 37.2 | 407.1 | 46.6 | 25.9 | 0 |
| tristantro | 2 | 40.4 | 392.3 | 55.0 | 25.1 | 1 |

Read it honestly: the model's edge (~+40 lineup pts a week over gilbobo) is measured with the
model's own valuation, so it is an upper bound. **The market rates this roster lowest and it is
the oldest by five years** -- LeBron 41, Durant 38, DeRozan 37, Lillard 36 (Achilles), Kawhi
35, Vucevic 35, Kyrie 34. Lock-In makes the odd rest day cheap; it does not protect against
long injuries, and the model has no age decline term. gilbobo (Luka, Booker, AD, Jalen
Johnson, Chet; median ADP 33) is the threat by both measures. tristantro drafted Kendrick
Perkins, a retired player.

**Incident: the advisor stopped at 18:31** during the real draft (page last written
18:31:08, pick 38 on the clock). Sleeper was reachable from the same machine, and
`build_view` ran cleanly on the exact live state (0.13 s), so it was not the data. The user
restarted it and it ran to the end. No traceback was captured, so the cause is still
unknown. Only the fetch was inside a `try`; the most plausible Windows candidates are
`os.replace` in `write_atomic` refusing while the browser reads the page, and a cp1252
`print` failing on a name like "Vučević" if output was a pipe. **Fixed 2026-10-21 (step 3):**
the whole poll now runs through `runtime.run_loop` -- a failed pass logs its traceback to
`data/logs/draft_live.log`, shows the error on the page, and the next pass runs. If it ever
happens again, that log has the cause.

### Measured after the mock, not yet acted on: rookies

Backtested on the 2024-25 and 2025-26 rookie classes (116 rookies, 30 with real ADP), using
only pre-season information, against actual fp/game:

| Method | r (all) | MAE | r (ADP subset) |
|---|---|---|---|
| current: cohort at projected minutes | 0.33 | 6.9 | 0.38 |
| preseason fp/min x projected minutes | 0.46 | 7.6 | 0.39 |
| blend, preseason rate shrunk to cohort (60-min prior) | 0.44 | 6.8 | 0.43 |
| Sleeper ADP alone (log-linear, out-of-season) | -- | -- | 0.39 |

All weak; minutes are the bottleneck (projected vs actual minutes r = 0.35) and every method
missed Flagg by ~15 fp. The blend is a small real improvement, ~30 min to implement with
tests, and was **deliberately not shipped before the draft**. It matters again only for
waiver decisions on rookies. Preseason logs for 2024-25 and 2025-26 are now on disk
(`preseason_logs_{season}.parquet`), fetched for this backtest.

**Never backtest against Sleeper's past "projections"**: the 2025 rows are actual 2025-26
output overwritten after the season (r = 1.000 for rookies). ADP in the same rows is genuinely
pre-season. Recorded in `engine/ingest/sleeper.py`.

### Phase 3 step 1 — week-1 lock cheat sheet (done 2026-10-19)

`uv run python -m engine.jobs.lock_sheet [--week N]` -> terminal + `data/lock/lock_sheet.html`
(self-contained, offline) and `lock_sheet_week{N}.html`. 12 fixture tests; four mutations
(B2B looking forward, threshold off by one, preseason counted, any B2B = trap) each fail it.

- For each rostered player: this week's games (date, opponent, home/away), the lock threshold
  after each game = `lockin.lock_threshold(dist, games_left_after_it)`, B2B second nights
  marked, and an **AUTO-LOCK TRAP** warning when his *last* game of the week is a B2B second
  night (the fallback game, 0 if he sits). Closed-form check pinned: normal(45, 10) over 3
  games gives thresholds 48.99, 45.0, then automatic.
- B2B = his team played the previous calendar day, looked up across the whole regular-season
  schedule, so a Monday after a Sunday in the previous week counts. Preseason is excluded.
- **Lineup:** recommends the nine starters maximising this week's expected lock-in value
  (`weekly_value` at his game count, exact assignment via `lineup.assign_lineup`) and lists
  the changes vs Sleeper's current lineup. Only starters can be locked. On 2026-10-19
  Sleeper had Lillard and Kyrie benched for Caleb Wilson and Vucevic; the sheet says swap.
- Week 1 findings: LeBron, Brown (both end Sun vs DET after Sat vs MIL), Durant, Bam and
  Kyrie all end the week on a B2B second night. Bam was Sleeper DTD.
- **`engine/model/player_model.py`** now holds the distribution recipe (minutes-band priors,
  overrides, `PlayerModel.veteran / rookie_at / distribution`), moved out of
  `build_board.main()` so board, sheet and nightly share one implementation. Verified: every
  player present before and after has identical values in every `pool.csv` column (three
  players disappeared only because they were waived overnight).
- **`SleeperClient.get_rosters_live`** -- `get_rosters` caches forever, which is wrong
  in-season. The user's roster: `owner_id` = his `user_id`, `roster_id` 4.
- Known limits: the schedule parquet has dates, not
  tip times (`normalize_schedule` drops them -- M6 needs them for real deadlines); Sleeper
  injury status is shown but does not change numbers; the lineup advice is weekly although
  Sleeper allows changes before each game.

### Phase 3 step 2 — back-to-back availability (wired in 2026-10-19, band ratio only)

Measured on 2023-26 logs, rotation players only (20+ min that season), 57,769 player-games,
using the same per-stint windows as `p_play`. Team schedules are reconstructed from the logs;
ages from Sleeper `birth_date` matched by normalized name (800 of 802 players; 7 duplicate
names dropped). Analysis script was ad hoc (scratchpad), not committed -- re-derive if needed.

| Age band | Plays, other nights | Plays, B2B 2nd night | Gap | Ratio |
|---|---|---|---|---|
| <30 (339 players) | 0.838 (n=37,314) | 0.796 (n=8,096) | -0.041 | 0.951 |
| 30-33 (73) | 0.835 (n=6,873) | 0.753 (n=1,514) | -0.082 | 0.901 |
| 34+ (27) | 0.867 (n=3,268) | 0.761 (n=704) | -0.105 | 0.879 |

- Real and age-graded; 95% intervals do not overlap in any band.
- **It is rest, not lingering injury:** conditional on playing the team's previous game,
  other nights are ~0.94 in every band and B2B second nights 0.894 / 0.865 / 0.825 (ratios
  0.951 / 0.923 / 0.881) -- the same picture.
- **Survivor effect:** 34+ rotation players are *more* available on ordinary nights (0.867).
  So this must be a multiplier on B2B nights, not a flat age penalty on `p_play`.
- The user's players, B2B second night vs other nights (24-46 B2B games each, noisy):
  Kyrie 0.58 vs 0.80, Kawhi 0.68 vs 0.87, LeBron 0.72 vs 0.90, Brown 0.74 vs 0.87, Lillard
  0.79 vs 0.89, Jokic 0.80 vs 0.89, Durant 0.88 vs 0.91, DeRozan 0.93 vs 0.96, Vucevic 0.93
  vs 0.86, Bam 0.89 vs 0.91, George 0.82 vs 0.86, Bey 0.88 vs 0.96.

**How it is wired.** The user chose the **band ratio only**; the per-player blend (shrinking
each player's own B2B ratio toward the band by sample size -- harsher on Kyrie/Kawhi/LeBron,
softer on Durant) is deferred to its own step.

- `distributions.py`: `B2B_AVAILABILITY_RATIO_BY_AGE` (the table above), `age_on`,
  `game_p_play(base, b2b_second, age)` = base x ratio on a B2B second night, base otherwise.
  Age is taken **on the game date** (Jaylen Brown turns 30 on 2026-10-24, so his Sunday game
  takes the 30-33 ratio). `age=None` leaves the game unpriced and the sheet flags it -- never a
  guessed band. Ages come from Sleeper `birth_date` (now on the `Player` model).
- `lockin.py`: `roll_on_values` / `weekly_value` / `lock_threshold` take an optional
  `p_play_by_game`; `None` is the old recursion exactly (regression green). For
  `lock_threshold` the list covers the *remaining* games only.
- `lock_sheet.py`: per-game `p_play` feeds the thresholds **and the week value**, so it also
  moves the lineup; B2B rows and the trap warning quote the priced chance he plays.
- Known approximation: base `p_play` already averages over B2B nights (~18% of games), so
  ordinary nights are understated by ~0.18 x (1 - ratio), at most ~2%. Not corrected.
- **The board and draft season values do not use it** (`season_value` still takes game counts
  only). Fine post-draft; worth adding when waiver ranking (step 7) needs per-week values.
- Tests: 24 new; 8 mutations each fail the suite (ratio on every night, recursion ignoring the
  list, final game ignoring its own p, list reversed, 34+ band at 35, threshold sliced at the
  game just played, age at week start, week value ignoring B2B).

Week 1 effect on the user's roster (same nine starters recommended before and after):
Kyrie's Sat B2B is now 67% to play, LeBron's Sun 76%, Brown's Sun 77% (he is 30 by then),
Durant's Sat 79%. Thresholds before those games fall 1-5 points (Durant Fri 44.1 -> 38.8,
LeBron Sat 38.7 -> 34.0, Kyrie Fri 34.8 -> 30.6); starting nine week value 482.9 -> 478.8.

### Phase 3 plan and status (set by the user 2026-10-19; stop after each step)

1. ~~Week-1 lock cheat sheet~~ -- **done** (above).
2. ~~Back-to-back `p_play`~~ -- **done**, band ratio only (above).
3. ~~Harden the poll loop~~ -- **done 2026-10-21.** `engine/jobs/runtime.py` holds
   `run_loop` (guards the whole pass, logs tracebacks to `data/logs/<job>.log`, hands the
   error to the job, keeps going; only `KeyboardInterrupt` stops it) and `utf8_output`
   (every job's `main()` calls it first, so `lock_sheet > out.txt` works). `draft_live`'s
   pass is now `LivePoller.poll`; `LivePoller.failed` puts the error on the page and keeps the
   last good view. Tested by raising inside `build_view` mid-draft; three mutations (loop
   re-raises, error not on the page, no UTF-8) each fail the suite. **The nightly job must
   use `run_loop`/`utf8_output` too.**
4. **M6 nightly advisor** (TASKS M6): probe the matchups endpoint for lock status first (rule
   6) -- if absent, infer it and add manual confirmation; `matchup.py` weekly win-probability
   sim vs this week's opponent; `nightly.py` LOCK/ROLL per started player with a one-line
   why, ordered by the earliest deadline, matchup-aware (roll more when behind, lock more
   when ahead). **Cloud-first, decided 2026-10-21:** runs on GitHub Actions cron, alerts go
   to a **Discord webhook** (`DISCORD_WEBHOOK_URL` in `.env` locally, a GitHub secret in
   Actions; never ntfy). In-season data avoids stats.nba.com (blocks cloud IPs):
   - box scores: Sleeper `api.sleeper.com/stats/nba/player/{id}?season_type=regular&season=2026&grouping=week`
     (per game, dated; probed 2026-10-21 -- Brown's 10/20 line rescored to Sleeper's 43.5)
     and the bulk `api.sleeper.com/stats/nba/2026?season_type=regular&week=N` (weekly totals);
   - schedule + game status: `api.sleeper.com/schedule/nba/regular/2026` (no tip times);
   - tip-off times: ESPN `site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD`;
   - 2023-26 history: static, uploaded once from the laptop (Supabase or similar); only the
     occasional history refresh stays local.
   Before relying on them: a one-off Actions workflow that probes every endpoint from GitHub's
   IPs (rule 6), and a local week-long diff of Sleeper box scores vs stats.nba.com. Actions
   cron can start 5-30 min late, so alerts fire well before tip-off, never at the deadline.

   **Status 2026-10-21 -- done:**
   - **Probe passed from GitHub Actions** (`probe_endpoints.yml`, run 39696604428): all 10
     required Sleeper/ESPN endpoints answer from GitHub's IPs. Whether stats.nba.com and
     cdn.nba.com do is unknown (step logs need a GitHub login); nothing depends on them.
   - **Sleeper's schedule = ours**: all 1200 games agree on date, teams and week; ours has 3
     extra TBD NBA Cup placeholders. So `nightly` uses Sleeper's schedule directly.
   - **Sleeper per-game stats carry `tf`/`ff`** (technicals, flagrants -- missing from
     stats.nba.com logs) and its own `dd`/`bonus_pt_*` keys. `scoring.box_from_sleeper_stats`
     converts; Brown 43.5 and LeBron 38.0 on 10/20 rescore to Sleeper's exact totals.
   - **No lock field in the public API** (matchups, rosters, roster settings, league all
     checked). `locks.infer_lock` infers it from `starters_points`: counting the latest = "open"
     (unlocked or locked tonight -- cannot be told apart until his next game); matching nothing =
     a visible gap. **No manual confirmation**: a cloud job cannot take input, so advice is
     phrased "only if you have NOT locked him".
   - **Corrected 2026-10-22: "counting an earlier game" is NOT proof of a lock after a game
     ends.** After the 76ers' Thursday game every 76ers starter in the league (LeBron, Brown,
     Embiid, Maxey; three rosters) still counted Tuesday's score; the user had locked neither of
     his, and LeBron's moved to Thursday's 43 within about an hour -- **Sleeper refreshes a
     starter's counted score some time after his game ends** (the bench player showed Thursday
     at once). Now: during a live game, stuck on an earlier game = **locked** (unlocked
     starters follow the live score, e.g. Jalen Williams); after a game, it is **"unclear"**.
     The user's own unclear player is advised as not locked, from his latest game (gap note
     says so); an opponent's is treated as banked at Sleeper's number. The bad rule had made
     the report call Brown and LeBron locked and drop their Saturday decisions.
     Week-1 ground truth (user): locked Durant 51, Lillard 49.5, Kawhi 48.5; not Brown, LeBron.
   - **Corrected again 2026-10-23: Sleeper's counted score never shows a lock.** Durant and
     Lillard (locked Wed) showed live scores in Friday's games; Kawhi (locked at 48.5) showed
     Friday's 34 after it. `starters_points` is each starter's latest game, refreshed late, lock
     or no lock -- the earlier "stuck during a live game = locked" rule was wrong too. Also seen:
     for ~an hour after a game, requests returned old and new scores inconsistently (CDN), and
     Brown still showed Tuesday's 43.5 a day after Thursday's 41. `locks.infer_lock` now never
     returns "locked"; **locks come only from the user, in `state/locks.json`** (Sleeper player
     id -> score, for one week; another week's file counts for nothing). When the user reports
     a lock, update it and push: the `state/**` path reruns the advisor. Opponents' locks are
     unknowable. With the four reported locks, week 1 win probability on Fri night was 53%,
     not the 33% the display-based report gave.
   - **Opponent assumed to lock optimally (user's choice, 2026-10-25).** `matchup.policy_lock`
     replays each opponent starter's finished games through the expected-value thresholds
     (per-game `p_play` over his whole week); the first game that clears its threshold is his
     lock, else the decision is still open. Ex-ante optimal can be ex-post poor: on Sun 10/25 it
     had Barnes locked at Wed 45 (Fri 60.5) and Herro at Wed 46 (Fri 62.5), and gave 92% to
     win; treating Sleeper's displayed numbers as banked had given 71%. The truth for a real
     manager is in between; both say lock George 39 and Brown 50.
   - **`matchup.py`**: simulates both lineups through the week (threshold policy, per-game
     B2B `p_play`, forced final game, 0 if he sits), common random numbers per slot, and
     `decide` = LOCK vs ROLL by chance of winning the week; falls back to the expected-points
     call within 2 standard errors. Reproduces M3's 51.297 by simulation. 5 mutations fail
     the suite.
   - **`nightly.py`** prints the report to the terminal. First live run (Wed 10/21, games in
     progress): 87% to win week 1 vs 2jake4davis6; Brown and LeBron ROLL, matchup agrees.
   **Also done, later 2026-10-21:**
   - **Discord** (`engine/jobs/discord.py`): `nightly --post` sends a short phone version
     (`render_discord`), split under Discord's 2000-char cap. The user created the webhook and
     the `DISCORD_WEBHOOK_URL` repository secret that evening; their local `.env` is not made
     yet, so local runs print "not posted". A failed run posts "Nightly advisor failed" and
     exits non-zero.
   - **Tip times** (`engine/ingest/espn.py`): ESPN abbreviates six teams differently (GS, NO,
     NY, SA, UTAH, WSH -> GSW, NOP, NYK, SAS, UTA, WAS); unknown codes are dropped, not
     guessed. Deadlines print in ET; decisions are ordered by tip-off.
   - **`snapshot/`** (committed, ~1.2 MB): `publish_snapshot` copies `pool.csv` and the game
     logs there for Actions. It is a copy, so the cloud model learns this season only when
     refreshed from the laptop; the report flags it once it is over 8 days old. Rebuilding
     current-season logs from Sleeper's per-game stats in the cloud would remove that chore.
   - **`.github/workflows/nightly.yml`**: 07:43 UTC full report, 19:43 UTC reminder only if a
     decision is due that day, plus manual runs. Both workflows on checkout v5 / setup-python v6.
   - **Re-run on model change** (user request, 2026-10-22, after the age adjustment flipped
     Kawhi from ROLL to LOCK with no explanation in Discord): any push touching
     `engine/model/**`, `engine/ingest/**`, `nightly.py`, `lock_sheet.py`, `build_board.py`,
     `config/league.json`, `config/player_overrides.csv` or `snapshot/**` reruns the advisor
     and posts a full report headed "Re-run after a model change: <commit subject>". The commit
     message reaches the script as an env var, never interpolated into the shell.
   - First real report (Wed night, week 1): 74% to win; **LOCK Durant 51 and Lillard 49.5**,
     roll the rest. All 17 finished games in the matchup rescored to Sleeper's exact counted
     scores -- early support for the scoring config, though none tested the open flags.
   - **Box-score diff done 2026-10-25** (week 1, every rostered player, 258 player-games):
     235 score identically from stats.nba.com logs and Sleeper's per-game stats; 20 differ
     only by technicals/flagrants (Sleeper has them, stats.nba.com does not -- the known
     `GAME_LOG_MISSING_STATS` gap); 3 differ by one steal (a stat correction on one side).
     **Sleeper's per-game stats are the better source**, so building current-season logs from
     them in the cloud is safe -- that would end the weekly laptop snapshot refresh.
   - **Failed run, Mon 2026-10-26** (schedule, started 14:34 UTC -- GitHub ran the 07:43 job
     ~7 h late): exit 1 after **32 s**, i.e. early, in a fetch; the same commit and snapshot ran
     cleanly afterwards, so it was transient. Step logs need a login and the public page showed
     only "exit code 1". Fixes: `engine/ingest/http_retry.py` -- every Sleeper, ESPN and Discord
     call retries twice (2 s, 6 s) on timeouts, connection errors, 5xx and 429, never on other
     4xx; and a failed run in Actions prints a `::error` annotation with the error and its
     innermost frame, which **is** public on the run page.
   - **Week-2 prep, 2026-10-25.** First snapshot refresh with this season's games (859
     regular-season rows). It exposed a bug: **every rookie fell out of the pool** once he had
     played (no longer preseason-only, not yet 10 games) -- 43 players, incl. the user's Caleb
     Wilson. `candidate_players` now also keeps anyone with a game in the season being
     played; pool back to 504, top-200 safety net empty. `state/locks.json` moved to week 2,
     empty. **Known weakness found:** `p_play` from a handful of games is not shrunk, so Caleb
     Wilson shows "plays 100%" and the week-2 lock sheet suggested starting him over LeBron --
     at a typical 87% the two are a toss-up. Shrink availability for thin histories next.
5. ~~Resolve the scoring ambiguities~~ -- **done 2026-10-22: both true.** `engine/jobs/
   resolve_scoring.py` used **last season's league** (identical scoring, Lock-In, complete)
   instead of waiting for week 1: 2666 player-weeks, each game rescored under all four flag
   combinations against Sleeper's counted `players_points`. 54 weeks could tell them apart
   (triple-doubles, 50-point games); only `bonuses_stack=true, td_includes_dd=true` matched
   all 54 (the old default matched 0, stack-only 8, dd-only 45). **0 of 2666 unexplained**,
   so the whole scoring engine is confirmed against Sleeper, not just the flags. A 50-point
   game pays 40+ and 50+ (7); a triple-double pays dd and td (7). Effect: Jokic +1.32
   fp/game (+2.58 VOR), Giddey +0.53, Doncic +0.42; 123 of 496 players move at all. Board
   and `snapshot/` rebuilt. Thursday calls unchanged (Brown 54.8, LeBron 49.4).
6. ~~Age + role awareness~~ -- **done 2026-10-22.**
   - **Measured as a backtest of the model, not raw year-over-year change** (raw ratios put
     every age band below 1.0: selecting by last season's minutes is regression to the mean).
     `PlayerModel` built from earlier seasons only, projected mean vs actual for 2024-25 and
     2025-26, 713 players with 20+ games in both seasons:
     `log(actual/projected)` = **−0.0176 per year of age** (se 0.0034; −0.0159 and −0.0195 in
     the two seasons separately). Minutes effect is U-shaped (×1.53 under 12 min, ×0.87-0.89
     at 18-30, ×0.98 over 30): role change, left to current-season data, not made a rule.
   - **35+ do not follow the line**: predicted ×0.84 vs 25-29, measured ×0.98 (95% 0.89-1.07,
     29 player-seasons). Role players collapsed (Conley, Lowry, Lopez, Gordon, Batum), stars
     held (Durant, Curry, Harden, LeBron at 40). **User's choice: the line, held flat from
     34** (×0.884 for everyone 34+).
   - `engine/model/aging.py`: factor relative to age 27 (reshapes, does not move the league
     level), age on 1 Feb of the season, **faded by the share of weight still from earlier
     seasons** so it disappears as this season's games arrive. Veterans only (the rookie
     cohort is calibrated separately); before overrides. `PlayerModel(season=, birth_dates=)`
     switches it on (board, lock sheet, nightly); without them the model is unchanged, which
     the backtest needs. No birth date -> unadjusted and listed, never guessed (0 today).
   - **Effect on the user's roster (big, on paper):** the seven 34+ players ×0.884; Jokic
     ×0.93, Brown ×0.95, Bam ×0.97, George ×1.07. VOR rank: Durant 9->26, Kawhi 13->32, LeBron
     24->44, Lillard 22->43, Kyrie 38->75; Jokic 1->3, George 42->21. Win probability week 1
     68% -> 54%. **Calls changed: Kawhi now LOCK at 48.5** (threshold 49.6 -> 43.9). Brown and
     LeBron still ROLL (52.0, 43.7).
   - **Role watch** (`engine/model/roles.py`, in `nightly`): minutes this season (Sleeper
     `min`, cloud-safe) vs his **most recent earlier season** (Lillard and Kyrie vs 2024-25),
     judged from 5 games, flagged at ±4 min; team change reported, not flagged by itself. Whole
     roster in the terminal report; Discord only gets flagged players.
   - **Found: the lock sheet's lineup ignores games played and locks.** With aging it said to
     bench Lillard, locked at 49.5 the night before. It now prints a caveat; mid-week the
     nightly report is the tool.
   - Tests: aging 7, roles 6; 6 mutations fail the suite (no flat tail, no fade, age today,
     guessed birth date, flag boundary, judged from game 1).
7. **Roster management -- in progress.** **Done 2026-10-22: IR alerts and pickups**, in
   `nightly` (terminal: always; Discord: only when there is something to do):
   - `engine/model/roster_moves.py` (pure). Sleeper `injury_status` values in use (probed):
     `Out`, `GTD`, `DTD`, `IR`. Out -> OUT (eligible); GTD/DTD not; Sleeper's own `IR`
     treated as eligible but flagged "assumed"; `Sus`/`Suspended`/`DNR` mapped with no live
     example; **anything else is reported as unrecognised**. Alerts: stash an eligible player
     while a slot is free, "both slots full", and **activate** a reserve player who no longer
     qualifies (Sleeper restricts moves while one sits there). A roster's `players` includes
     its `reserve`; active = players minus reserve.
   - Pickups: free agents (on an NBA team, on no roster) whose board VOR beats the user's
     weakest **bench** player by `PICKUP_MIN_GAIN` = 2.0 per week; OUT/IR free agents skipped.
     **Waivers inferred, not read**: dropped within `clear_days` (from transactions), or his
     game today has tipped (`after_game_start`). The one processed claim so far ran Wed
     03:08 ET (just after midnight PT), so the 07:43 UTC report lands after waivers clear.
     Shows the user's `waiver_position`.
   - First live run (Thu 10/22): IR nothing to do; you are #7 in priority; five free agents
     beat Vucevic (-3.6 VOR after aging) by 3-5/week: Jeremiah Fears, Jabari Smith Jr. (one
     of the 32 the suffix bug had hidden), Kyle Filipowski, Bennett Stirtz, Ace Bailey. All
     verified unrostered.
   - Tests: roster_moves 10, nightly_roster 5; mutations (rostered player offered, IR player
     counted active) fail the suite -- the second only after adding a still-out IR case.
   **Also done 2026-10-22: trade finder and deadline countdown** (`engine/model/trades.py`,
   in `nightly`):
   - Team strength = best starting lineup (exact assignment) valued by `season_value`. A trade
     of up to 2 players each way is listed only if it raises **both** lineups by `MIN_GAIN` =
     1.0/week; each side shows Sleeper's rank (`search_rank`, the market view the partner
     sees). Throw-ins that change nothing are dropped (Kawhi + LeBron for Haliburton was also
     listed with Jarrett Allen added, same gains). All 7 partners searched in ~4 s.
   - First live run: user has the best lineup by our model even after aging (453.6/week;
     gilbobo 444.9, ChickenJoe2107 443.3), so mutual trades are rare and small (+1.3-1.7/week):
     Kawhi + LeBron for Haliburton (tristantro), Kyrie + Bey for Jaren Jackson Jr.
     (tristantro), Kyrie + LeBron for Harden (cyapp1), Kawhi + Bey for Michael Porter Jr.
     (cyapp1). Limits (docstring): lineup value only, no bench-depth value; our values for both
     sides.
   - Deadline = last game day of `trade_deadline_week` in Sleeper's schedule: **Sun
     2027-02-28**. Terminal shows trades every run; Discord only on Mondays, and daily in the
     deadline's last 2 weeks.
   - **The "sell an old player before the week-22 playoffs" flag was measured and dropped.**
     Fantasy playoffs are weeks 22-24 = Mar 15 - Apr 4. Rotation players at end of February,
     2023-26 (661 player-seasons): everyone plays less in that window (0.85-0.86 of team games
     before March -> 0.72-0.79), and **34+ drop the least** (ratio 0.925 vs 0.84-0.86 for every
     younger band); "sat the whole window" is ~10-11% in every band. Durant, LeBron, DeRozan
     played 90-100% of last season's window. Age decline is already in `aging.py`.
     **Worth wiring later: the ~15% league-wide late-season availability drop** lowers lock
     thresholds in weeks 22-24 for everyone.
   - Tests: trades 8 (+1 in nightly_roster); mutations (one-sided trades allowed, throw-ins
     kept) fail the suite.
   - **Rookie blend: skipped (user's choice, 2026-10-22).** It only applies until a rookie has
     regular-season games in the snapshot (weeks), and the measured gain was small (r 0.33 ->
     0.44, MAE 6.9 -> 6.8). A weekly snapshot refresh brings rookies' real games in sooner.
   **Step 7 is done.** Open follow-ups from it: the late-season availability drop for weeks
   22-24 (above), and rebuilding current-season logs from Sleeper in the cloud so the
   snapshot refresh stops being a laptop chore.

   Original plan: free-agent ranker (this-week and rest-of-season
   value vs the user's lowest-value player, through `player_model`); with reverse-standings
   priority (no FAAB) a claim costs nothing later, so the question is only whether a
   higher-priority team will take the player first, else wait a day for free agency; IR alerts when a
   rostered player turns OUT/SUS/DNR, so he can be stashed and a replacement added; trade finder seeded with the forward surplus (Brown, Durant, Kawhi, LeBron,
   DeRozan, Bey) vs other rosters' needs, a week-19 deadline countdown and a "sell an old
   player before the week-22 playoffs" flag; fold the measured rookie blend into `rookies.py`.

At the end: update this file's "Current state" and plan, the README, and fix `docs/SPEC.md`
(§7.2 IR stays but with the OUT/SUS/DNR eligibility; §7.3 priority waivers, no FAAB; two
divisions).

## Stack

- **Engine:** Python 3.11+, `uv` for deps. numpy, pandas, scipy, httpx, pydantic, pytest.
- **Storage:** Supabase (Postgres). Migrations in `supabase/migrations/`.
- **Web:** React + Vite + TypeScript + Tailwind, deployed on Vercel at `fantasy.ferf.gg`
  (DNS in Route 53, same pattern as the househunt tracker).
- **Scheduling:** GitHub Actions cron for anything that talks to Sleeper/Kalshi/Polymarket.
- **Notifications:** Discord webhook (`DISCORD_WEBHOOK_URL`, a secret) for lock, waiver and IR
  alerts. Chosen over ntfy 2026-10-21.
- **LLM:** Anthropic API, optional. Must degrade to a "copy this briefing into a chat" text
  blob if no key is set. See `docs/SPEC.md` §8.

## Known environment traps

- **stats.nba.com blocks datacenter IPs.** Anything using `nba_api` will work locally and
  likely fail on GitHub-hosted runners. Design ingestion as a local-first script that writes
  parquet to Supabase, and keep it runnable by hand. Do not silently retry into a rate limit.
- **Player identity differs across sources.** Sleeper IDs, NBA.com IDs, and the free-text
  names in Kalshi/Polymarket markets do not agree. There is a mapping table with manual
  overrides; see spec §6.3. Unmapped players are a visible gap, never a dropped row.
- **Prediction market coverage is thin and uneven.** Most nights only a handful of players have
  props. Treat market data as a bonus signal that improves a projection when present, never as
  a required input.
- **Windows Smart App Control is ON in enforcement mode on the dev machine.** It blocks
  compiled `.pyd`/`.dll` files that are neither signed nor known-good to Microsoft's
  reputation service, and the failure looks like
  `ImportError: DLL load failed ... An Application Control policy has blocked this file`.
  Reputation is **per file hash**, so a brand-new release of a package can be blocked while
  the previous release and every other package load fine — that is exactly what happened with
  pandas 3.0.6 (blocked) vs 2.3.3 (fine), which is why `pandas` is pinned `<3` in
  `pyproject.toml`. If a new dependency fails to import this way, check
  `Microsoft-Windows-CodeIntegrity/Operational` in Event Viewer for the verdict, then try a
  slightly older release before anything drastic. **Do not suggest turning Smart App Control
  off:** it cannot be turned back on without reinstalling Windows.
  - Known-blocked and accepted: `_hashlib.pyd` in uv's own unsigned CPython build
    (`AppData\Roaming\uv\python\...`). Cosmetic — `hashlib` falls back to Python's builtin
    SHA implementations, `_ssl.pyd` loads, and HTTPS to Sleeper works. It only produces a
    Windows toast. The clean fix, if it ever becomes worth it, is a python.org 3.11 install
    (PSF-signed) plus `tool.uv.python-preference = "only-system"`.

## Conventions

- Type hints everywhere; pydantic models at every I/O boundary.
- Pure functions for anything mathematical, so it can be tested without network access.
- Cache all external API responses to disk with timestamps; never hammer an endpoint in a loop.
- Every fantasy-point calculation flows through `engine/model/scoring.py`. One implementation.
- Commit messages: `phaseN: <what>` for the phase the work belongs to (`phase3:` for
  in-season work). Keep phases separable, see `docs/TASKS.md`.

## Testing

- Golden tests: hand-scored box scores in `tests/fixtures/` verify `scoring.py` including the
  bonus edge cases and both branches of the ambiguity flags.
- The optimal-stopping module has a closed-form regression test: for a normally distributed
  player with mean 45 and sd 10, optimal weekly lock-in value is ≈45.0 / 49.0 / 51.3 / 52.9
  for 1 / 2 / 3 / 4 game weeks. Simulation must reproduce this within tolerance.
- No test may require network access. Fixtures only.

## What "done" looks like for the user

He is on his phone, mid-draft, with 90 seconds on the clock. He needs one screen: who to take,
how likely each is to survive to his next pick, and one line of why. Everything else is
secondary to that.
