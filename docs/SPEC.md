# Fantasy Lock-In Assistant — Technical Spec

Target league: 8-team Sleeper NBA, Lock-In mode, snake draft. Full rules in `config/league.json`.
Season opens 2026-10-20. Draft date TBD, assume it can be called with two days' notice.

---

## 1. Repo layout

```
CLAUDE.md
README.md
config/
  league.json               # source of truth for rules
  player_overrides.csv      # manual name→id mappings and projection adjustments
engine/
  ingest/
    sleeper.py              # league, rosters, matchups, players, draft, picks
    nba_stats.py            # game logs, schedule  (RUN LOCALLY, see CLAUDE.md)
    markets.py              # Kalshi + Polymarket
    news.py                 # injury/status feed
  model/
    scoring.py              # box score → fantasy points
    distributions.py        # per-player single-game score distribution
    lockin.py               # optimal stopping / weekly value
    valuation.py            # season value, replacement level, VOR
    lineup.py               # slot assignment (bipartite matching)
    draft_sim.py            # Monte Carlo remainder-of-draft
    matchup.py              # weekly win probability, lock-or-roll decisions
  jobs/
    build_board.py          # phase 1 output
    draft_live.py           # phase 2 loop
    nightly.py              # phase 3 job
  tests/
web/                        # Vite + React + TS
supabase/migrations/
.github/workflows/
```

---

## 2. Data sources

Verify each endpoint before coding against it.

### 2.1 Sleeper (free, no auth, read-only)
Base `https://api.sleeper.app/v1`. Stay well under 1000 requests/minute.

- `GET /league/{league_id}` — scoring_settings, roster_positions, settings
- `GET /league/{league_id}/rosters` — roster → owner, players, starters
- `GET /league/{league_id}/users` — display names
- `GET /league/{league_id}/matchups/{week}` — points, starters_points, players_points
- `GET /league/{league_id}/drafts` and `GET /draft/{draft_id}` — draft metadata, draft_order
- `GET /draft/{draft_id}/picks` — picks as they happen (poll target for phase 2)
- `GET /players/nba` — full player dictionary incl. positions and injury_status.
  Large payload; fetch at most once per day and cache.

Lock status may not be exposed in the matchups payload. Probe it early. If it is absent, infer
locked slots by watching whether a starter's points stop changing after his game ends, and let
the user confirm in the UI.

### 2.2 NBA statistics (free)
- `nba_api` → `PlayerGameLogs` for per-game box scores (previous 3 seasons), `CommonTeamRoster`,
  team schedule. Blocked from cloud IPs; run locally.
- Static season schedule JSON from cdn.nba.com is a lighter-weight alternative for the
  schedule grid, which is all phase 1 needs beyond game logs.
- Preseason box scores are useful in October for detecting new roles.

### 2.3 Prediction markets (free read access)
- **Kalshi:** NBA player props exist for points, rebounds, assists, threes, steals and
  double-doubles, plus season awards. Coverage per game is limited, often a handful of players.
- **Polymarket:** public Gamma/CLOB read APIs, game markets and some props.

Use them for: (a) implied distributions for tonight's and this week's games, (b) an early
signal that a player is sitting — props pulled or a line dropping sharply usually precedes
official injury news.

Convert prices to probabilities, remove the vig, and fit a distribution to whatever thresholds
exist. With one or two thresholds, fit only a location shift on the player's historical shape.
Do not attempt a full nonparametric fit from three data points.

---

## 3. Lock-In mechanics (read twice)

Only **one game per player per week** counts. After a player's game ends, the manager may lock
that score; it must be locked **before that player's next game tips off**. A player must have
been in the starting lineup when the game began. Bench players cannot be locked. If nothing is
locked, Sleeper takes the player's team's **final game of the week**, which is **0 if he did not
play**, and never reverts to an earlier game. Locks are irreversible and freeze the slot.

Consequences the whole model rests on:

- Value is not season averages. It is the expected value of an optimal stopping rule over the
  games a player has in a week.
- **Volatility is an asset.** Two players with equal means are not equal; the higher-variance
  one is worth more, because only the kept game counts.
- **Games per week still matter**, but far less than in traditional formats. More games are
  more draws from the distribution.
- **Missed games are cheap.** Rest-prone and injury-prone stars are systematically underpriced
  by drafters using standard rankings.
- **The auto-lock fallback is the main way to lose points.** Never hold an unlocked star into
  his final game of the week without a reason.

---

## 4. Scoring engine (`scoring.py`)

Inputs: a box score row. Output: fantasy points.

```
fp = 1.5*pts + 1.0*reb + 1.0*ast + 1.0*stl + 1.0*blk - 1.0*tov + 0.5*fg3m
   - 1.0*tech - 2.0*flagrant
   + dd_bonus + td_bonus + pts_bonus
```

Double-double and triple-double are computed from the five countable categories
(pts, reb, ast, stl, blk) at ≥10 each. Bonus stacking and whether a triple-double also pays the
double-double bonus are **unresolved**; both are flags in `config/league.json`. Implement both
branches and expose which is active in the UI, because it changes the value of Jokić-type
players materially.

Sanity check at build time: recompute a handful of week-1 Sleeper box scores from raw stats and
diff against Sleeper's own totals. That resolves the ambiguity flags empirically.

---

## 5. Valuation engine (phase 1)

### 5.1 Per-game distribution (`distributions.py`)
For each player, build a distribution of single-game fantasy points:

1. Rescore the last 1–3 seasons of game logs with `scoring.py`.
2. Weight recent games more heavily (exponential decay, half-life ≈ 25 games).
3. Shrink toward a prior built from players with similar role and minutes. Rookies and
   role-changers have no usable history; they need a projection-based prior plus a manual
   override hook.
4. Model separately: `p_play` (probability of appearing in a given game) and the conditional
   score distribution given he plays. Rest patterns and injury status feed `p_play`.
5. Represent the conditional distribution empirically (the weighted sample) rather than
   assuming normality. The right tail matters because of the 40+/50+ bonuses.

Adjustments for the huge 2026 offseason player movement (LeBron to Philadelphia among others):
historical distributions misjudge anyone whose role changed. Support a per-player multiplier on
usage/minutes in `config/player_overrides.csv`, and revisit after preseason.

### 5.2 Weekly lock-in value (`lockin.py`)
For a week with games `1..n`, define by backward induction the value of entering game `i` with
nothing locked yet:

```
W_{n+1} = 0                                    # week over, nothing locked
W_n     = E[X_n]                               # forced: final game, 0 if DNP
W_i     = p_play * E[max(X_i, W_{i+1})] + (1 - p_play) * W_{i+1}
```

The optimal rule is: **lock game i iff the observed score exceeds `W_{i+1}`**. That threshold is
exactly what the nightly job reports in phase 3.

Weekly value for a player = `W_1` for that week's game count. Season value = mean over the
season's weeks, using the real schedule.

Regression test (normal, mean 45, sd 10, p_play = 1):
`W ≈ 45.0, 49.0, 51.3, 52.9` for n = 1, 2, 3, 4.

### 5.3 Replacement level and VOR (`valuation.py`)
8 teams × 9 starters = 72 starting slots; 104 rostered. Replacement level is **slot-specific**,
because the roster requires 2 C (16 leaguewide) plus flex G/F/UTIL.

Compute replacement by simulating a greedy league-wide draft of the top players into the slot
structure, then taking the marginal player at each slot type. Do not use a flat "player #72"
cutoff; it undervalues centers.

Value over replacement = season lock-in value − slot replacement value.

Expected output of phase 1: `board.csv` / a web table, sorted by VOR, with columns for weekly
value, volatility, games-per-week profile, p_play, slot eligibility, and a flag for players
whose history is unreliable.

### 5.4 Lineup assignment (`lineup.py`)
Given a roster and per-player values, assign players to PG/SG/G/SF/PF/F/C/C/UTIL optimally.
This is a max-weight bipartite matching; use `scipy.optimize.linear_sum_assignment`. Needed for
roster valuation in the draft simulator and for weekly lineup advice.

---

## 6. Draft assistant (phase 2)

### 6.1 Loop
Poll `GET /draft/{draft_id}/picks` every 3–5 seconds. On any new pick: update the pool and each
team's filled slots, then recompute. 104 total picks, 7–14 picks between the user's turns — a
full Monte Carlo between picks is affordable.

### 6.2 Simulation
For each candidate player, run N ≈ 2000 simulations of the remainder of the draft:

- **Opponent model:** each opponent picks from a softmax over `-(ADP_rank)` plus a positional
  need bonus for unfilled slots, with Gumbel noise. Start all opponents on a generic prior and
  **update per-coach during the draft** from observed picks (e.g. two early bigs → tilt toward
  bigs). Show the confidence of each opponent model in the UI.
- **Own policy:** greedy VOR with slot constraints for the remainder.
- **Outputs per candidate:**
  - `EV_roster`: expected value of the user's final starting lineup (via `lineup.py`) if he
    takes this player now.
  - `p_survive`: probability the player is still available at the user's next pick.
  - `threat[team]`: each opponent's projected final lineup value; the playoff cut is top 4 of 8,
    so the teams near that line matter most.
  - `denial`: threat-weighted drop imposed on opponents picking before the user's next turn,
    i.e. (value of this player to them) − (value of their next-best alternative).

### 6.3 Blocking policy
Denial is a **tiebreaker, not a co-objective.** Rationale: taking a player to deny an opponent
costs the full gap between him and the user's best available, while the harm inflicted is only
the gap between him and the opponent's next-best option — usually small in a shallow 8-team
pool, and shared across seven opponents while all of them benefit from the user's weaker pick.

Implement as: choose by `EV_roster`; when candidates are within a margin `ε` of the leader,
break by `denial`. Expose `ε` as a live slider (default small, ~1–2% of EV).

Two exceptions where denial is genuinely large and should be surfaced prominently:
- **Centers** (16 slots leaguewide, real cliff).
- **Reliable double-double bigs** (+3 bonus concentrates value in a small group).

Early-round threat estimates are noisy and should be labelled as such; they only become
meaningful around round 6, by which point blocking matters least.

### 6.4 UI
One mobile screen: candidate rows with `EV_roster`, `p_survive`, `denial`, the opponent it
hurts, and a one-line rationale. Plus a fallback static board export in case the live poll
breaks on draft night. **The static board must work with no network.**

---

## 7. In-season engine (phase 3)

### 7.1 Nightly job
Runs after the last game of the night ends (US/Mountain). For every unlocked starter:

1. Recompute `W_{i+1}` for his remaining games this week, using current `p_play`, opponent
   defense, and market-implied adjustments where available.
2. Compare tonight's actual score to that threshold.
3. **Then adjust for the matchup, which overrides the naive threshold.** The objective is
   P(win), not expected points. Simulate the remainder of both rosters:
   - Comfortably ahead → lock conservatively, cut variance.
   - Behind → roll unlocked slots, seek variance.
4. Force a lock recommendation whenever the remaining schedule is risky: the player's next game
   is his last of the week, he is questionable, or he is on a minutes restriction. The
   auto-lock fallback pays 0 if he sits.
5. Push a notification with the specific actions, ranked by deadline (whose next game tips
   first).

### 7.2 Injury handling
- `p_play` folds injury designations from Sleeper's player dictionary and news into every
  decision.
- Market signals (props pulled, sharp line moves) are the earliest available indicator.
- 2 IR slots: flag draftable and waiver-available injured stars whose Sleeper designation makes
  them IR-eligible, so they cost no active roster spot.
- Buy-low window: with the trade deadline at week 19 and playoffs starting week 22, flag injured
  stars projected to return weeks 20–21 automatically.

### 7.3 Waivers
1-day waivers in a shallow league mean the wire is a real asset. Rank free agents by expected
contribution to *this week's* unlocked slots (games remaining matters more than season value),
and separately by rest-of-season VOR.

---

## 8. LLM layer (phase 4)

The model is a **writer and a reader, never a calculator.** Inputs: computed numbers, roster
states, league rules, opponent tendencies, raw news text. Outputs:

- One-line rationales on the draft board and nightly alerts.
- Parsing unstructured news (beat writer notes, coach quotes) into `p_play` and role adjustments.
- Trade proposals: given each opponent's roster needs and the threat ranking, draft the message.
- A weekly matchup briefing.

Cost is small (a few dollars a season). Two zero-cost fallbacks, both must work:
1. Emit a markdown briefing the user pastes into a Claude chat manually.
2. Local model on the user's RTX 5090 via Ollama, accepting weaker reasoning.

---

## 9. Open questions to resolve before the draft

1. League ID and the user's draft slot.
2. Draft date and time.
3. Do the 40+/50+ bonuses stack? Does a triple-double also pay the double-double bonus?
4. Does Sleeper's matchups payload expose lock status?
5. Weekly add/drop limits, if any (not visible in the settings screenshots).
6. Are Sleeper mock drafts reachable via the same draft endpoints? If so, use one as a phase 2
   dress rehearsal; if not, replay a completed draft from last season.
