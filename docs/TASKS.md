# TASKS

Ordered milestones. Each has an acceptance test. Do not start a phase before the previous one
passes, except M0 which unblocks everything.

Hard deadline context: NBA season opens 2026-10-20; the draft will be some days before that and
the date is not yet known. **Phase 1 must be finishable without a live draft connection**, so
that a static board is ready even if phase 2 is not.

---

## M0 — Bootstrap (do first, ~1 hour)

- [ ] `uv init`, Python 3.11, deps: numpy, pandas, scipy, httpx, pydantic, pytest, python-dotenv.
- [ ] Copy `CLAUDE.md`, `docs/SPEC.md`, `docs/TASKS.md`, `config/league.json` into place.
- [ ] `.env.example` with `SLEEPER_LEAGUE_ID`, `SUPABASE_URL`, `SUPABASE_KEY`,
      `ANTHROPIC_API_KEY` (optional), `NTFY_TOPIC`.
- [ ] `engine/ingest/sleeper.py`: fetch league, assert `config/league.json` matches
      `scoring_settings` + `roster_positions`, print a diff and exit non-zero on mismatch.

**Accept:** `python -m engine.jobs.verify_league` prints the league name, 8 teams, the roster
slots, and confirms scoring matches config.

---

## M1 — Scoring engine

- [ ] `scoring.py` implementing §4 of the spec, both ambiguity branches behind flags.
- [ ] Fixtures: 8–10 hand-scored box scores covering a plain line, a double-double, a
      triple-double, a 40-point game, a 50-point game, and a tech.

**Accept:** `pytest tests/test_scoring.py` green, including both settings of each flag.

---

## M2 — Historical data ingest (run locally)

- [ ] `nba_stats.py`: pull 3 seasons of player game logs and the 2026-27 schedule.
- [ ] Persist to parquet + Supabase. Re-runnable and incremental.
- [ ] Document the cloud-IP block in the README.

**Accept:** a local parquet with ≥3 seasons of logs; a schedule table giving games per team per
fantasy week for the whole season.

---

## M3 — Distributions and lock-in value

- [ ] `distributions.py` per §5.1, including `p_play` split out.
- [ ] `lockin.py` backward induction per §5.2.
- [ ] Regression test: normal(45, 10) → W ≈ 45.0 / 49.0 / 51.3 / 52.9 for n = 1..4.

**Accept:** for any player, the tool prints weekly value by game count and season value on the
real schedule. Spot check: high-variance scorers should rank above equal-mean steady producers,
and a known rest-prone star should lose only a few points of value.

---

## M4 — Board (phase 1 deliverable) 🎯 *ship before the draft*

- [ ] `lineup.py` slot assignment.
- [ ] `valuation.py` slot-specific replacement level and VOR.
- [ ] `build_board.py` → `board.csv` + a static HTML page that works offline.
- [ ] `player_overrides.csv` wired in for role changes (heavy 2026 offseason movement).

**Accept:** a ranked top-120 board. Eyeball test: centers with reliable double-doubles rank
above where standard fantasy rankings put them; defensive specialists rank below (steals and
blocks are only 1 point here); high-usage scorers are boosted by the 1.5 multiplier and the
40+/50+ bonuses.

---

## M5 — Live draft assistant (phase 2)

- [ ] Draft pick polling with resilient retry and a visible "last updated" clock.
- [ ] `draft_sim.py`: opponent models, per-coach in-draft updating, N≈2000 sims.
- [ ] Outputs `EV_roster`, `p_survive`, `threat[team]`, `denial` per §6.2.
- [ ] Blocking as tiebreaker with an ε slider per §6.3.
- [ ] Mobile screen per §6.4 + offline fallback to the M4 board.

**Accept:** a full dress rehearsal against a mock or replayed draft, producing a recommendation
in under 10 seconds per pick, and degrading to the static board when the network is pulled.

---

## M6 — Nightly lock advisor (phase 3, week 1 of season)

- [ ] `matchup.py`: weekly win probability simulation for both rosters.
- [ ] `nightly.py`: thresholds, matchup-aware variance adjustment, forced locks on risk.
- [ ] ntfy push with actions ordered by which lock deadline comes first.
- [ ] Probe whether lock status is exposed by the API; if not, add manual confirmation in the UI.

**Accept:** one week of live operation where every recommendation arrives before the relevant
deadline and no starter is ever left to the auto-lock fallback unintentionally.

---

## M7 — Markets integration

- [ ] Kalshi + Polymarket clients, cached.
- [ ] Player identity mapping with manual overrides; unmapped players visible, never dropped.
- [ ] Devig, fit a location shift onto the historical shape, feed into `p_play` and projections.

**Accept:** on a night with prop coverage, projections shift sensibly; on a night without, the
system runs unchanged. Pulled props raise a "possible rest" flag.

---

## M8 — LLM coach + trades (phase 4)

- [ ] Rationale generation on board and nightly alerts.
- [ ] News parsing into `p_play` / role adjustments.
- [ ] Trade finder against opponent rosters, week 19 deadline countdown, injured-star buy-low
      window for weeks 20–21 returns.
- [ ] Both zero-cost fallbacks working (paste-into-chat briefing; local model).

**Accept:** a weekly briefing that a human would act on, with every number traceable to the
deterministic engine.
