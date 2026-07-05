# Code walkthrough — how every module actually works

Companion to `how-it-works.md` (concepts). This is the code-level defense.

## Map

| File | Role | Key entry points |
|---|---|---|
| `kaplan_meier.py` | KM estimator from scratch | `KaplanMeier.fit`, `.median`, `.survival_at` |
| `logrank.py` | two-sample log-rank from scratch | `logrank_test` |
| `drawdowns.py` | prices → censored episodes | `extract_episodes`, `episodes_for_universe` |
| `designwins.py` | announcement corpus pipeline | `load_designwins`, `to_survival_table` |
| `data.py` | cached price downloads | `load_prices` |
| (lifelines) | Cox regression | used in the notebook with `cluster_col` |

## `kaplan_meier.py` — the estimator

`fit(durations, events)` mechanics:

1. **Stable argsort** by duration (`kind="stable"`): with tied durations, a stable
   sort makes the traversal order — and therefore the output — byte-identical across
   runs. Determinism is a feature, not an accident.
2. **One pass with a tie-grouping while-loop.** At index i, everyone from i onward
   has duration ≥ t, so the risk set is simply `n_at_risk = n − i` — that's why the
   array is sorted. The inner while-loop consumes *all* rows with duration == t,
   counting events d among them (censored rows at t leave the risk set without
   contributing an event — censoring handled in one line).
3. **Update only at event times**: `S *= (1 − d/n_at_risk)` when d > 0. Pure-censor
   times never append a curve point (the curve is a step function that only steps at
   events — a common implementation mistake is stepping at censor times too).
4. **Greenwood running sum**: `greenwood_sum += d/(n(n−d))`; the SE at each event
   time is `S · sqrt(greenwood_sum)`. Guard `n_at_risk > d` avoids 0-division when
   the last subjects all die at once.

Lookups: `survival_at(t)` = `searchsorted(event_times, t, side="right") − 1` — the
step-function convention that S(t) at an event time *includes* that event's drop;
index −1 (before the first event) returns 1.0. `median()` = first event time with
S ≤ 0.5, NaN if the curve never reaches it (all-censored data — a test covers it).
`confidence_band` clips plain Greenwood ± 1.96·SE into [0,1] (lifelines defaults to
log(−log) intervals — ours are cross-checked on the *curve and median*, which match
lifelines to 1e-10; the band convention difference is known and stated).

## `logrank.py` — the test

For each distinct event time t (union of both groups' event times):
`n_a, n_b` = counts with duration ≥ t (still at risk); `d_a, d_b` = events exactly
at t. Under H0 (shared hazard), the events at t are a hypergeometric draw:
expected `d·n_a/n`, variance `d·(n_a/n)·(n_b/n)·(n−d)/(n−1)` — the `(n−d)/(n−1)`
factor is the finite-population correction interviewers like to probe. Sum
observed−expected and variances over event times; statistic = (Σ O−E)²/ΣV ~ χ²(1);
p = `chi2.sf`. Skips times where either group has an empty risk set. Matches
lifelines' statistic and p to 1e-9 on randomised inputs.

## `drawdowns.py` — the episode state machine

A single O(n) pass holding three pieces of state: `in_episode`, `entry_high` (the
reference high **frozen at entry**), and running `depth`.

- **Entry**: first close ≤ (1 − threshold) × trailing `window`-day high
  (`rolling(window, min_periods=1).max()` — min_periods=1 so early history has a
  reference).
- **During**: `depth = min(depth, price/entry_high − 1)` — deeper drops extend the
  *same* episode; there is no re-triggering.
- **Exit (the event)**: price ≥ `entry_high` — the high in force at entry, not
  today's rolling high (recovery means regaining what you lost, not clearing a bar
  that fell while you were down).
- **Censoring**: still in an episode at the last row → `recovered=False`,
  `exit_date=NaT`, duration = days to sample end.

Baseline covariates are frozen **at entry**: `trailing_vol` (60-day realised,
annualised, `ddof=1`) and `market_dd` (SPY's own distance from its running peak that
day, reindexed + ffilled to the ticker's calendar). `max_depth` is also recorded —
deliberately, as the bait for the post-baseline trap in the notebook; the docstring
says never to use it as a covariate and the notebook shows why.

## `designwins.py` — the corpus pipeline

`load_designwins` validates loudly: required columns present, dates parseable,
segments from a fixed vocabulary, no blank tickers, and `revenue_event_date <
announce_date` rejected row-by-index. `to_survival_table(corpus, as_of)` builds the
(duration, event) rows: observed wins → days from announcement to revenue event;
unobserved → censored at `as_of`; `as_of` earlier than an announcement raises. The
output shape is exactly `KaplanMeier.fit`'s input — the whole point: when the
hand-collected corpus arrives, zero analysis code changes.

## Cox in the notebook — why those arguments

`CoxPHFitter().fit(df, duration_col, event_col, cluster_col="ticker")`:
`cluster_col` requests robust (sandwich) standard errors grouped by ticker, because
episodes across the five names share calendar-time shocks (2008 opens an episode in
all five at once) — 162 episodes are not 162 independent observations. The trap
model adds `max_depth` and is fit *identically*, so the only difference on display
is the covariate set.

## The tests, as a defense layer

- KM: a fully hand-worked 5-observation example (S = 0.75, 0.5, 0 at t = 2, 3, 5);
  curve and median vs lifelines on random censored data (3 seeds); all-censored →
  flat curve, NaN median; validation errors.
- Log-rank: exact match to lifelines (4 seeds); identical groups → statistic 0,
  p = 1; 10x hazard separation → p < 1e-10.
- Drawdowns: constructed paths with hand-known entries, exits, durations, depths;
  censored path; deeper-drop-same-episode; two separate episodes; covariate measured
  at entry (market 8% off its peak → `market_dd = 0.08`); rolling-window reference
  forgetting old highs.
- Design-wins: schema violations each raise with a matching message; survival-table
  round trip with exact day counts (320 observed, 751 censored).

## Grilling Q&A (implementation level)

- *Why is the risk set just n − i?* Because the array is sorted by duration:
  everyone at or after position i survives at least to t. That's the entire reason
  for the sort.
- *How does KM "use" a censored observation?* It sits in the denominator (n_at_risk)
  for every event time up to its censor time, then vanishes. It contributes exposure,
  never an event. Deleting it instead biases survival *down*... deleting long
  censored subjects biases the estimated durations down specifically because the
  unfinished episodes are disproportionately the longest.
- *Why `side="right"` in searchsorted?* So `survival_at(t)` at an exact event time
  returns the post-drop value — the right-continuous step convention.
- *Why does the log-rank variance have (n−d)/(n−1)?* Sampling d events *without
  replacement* from n at risk is hypergeometric, not binomial; the factor is the
  finite-population correction (it vanishes when d = 1).
- *Why not implement Cox from scratch too?* Partial likelihood with ties (Breslow/
  Efron), clustered sandwich errors, and convergence diagnostics is a project of its
  own; the from-scratch budget went to KM and log-rank where hand-verification is
  feasible, and lifelines' Cox is the cross-checked industry standard.
- *Why trading days, not calendar days?* Durations count bars between entry and
  recovery rows. Calendar days would mix weekends into the units; either is fine if
  stated — consistency is what matters.
