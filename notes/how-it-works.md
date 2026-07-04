# How this project works — study notes

Plain-English walkthrough. Read alongside `notebooks/study.ipynb`.

## What this project is

Survival analysis — the statistics of *time until an event*, with incomplete
observation — applied to semiconductor equities. The motivating question is design-win
revenue timing (how long from announcement to revenue ramp), which needs hand-collected
data; the executed study validates every piece of machinery on a question free data
supports: how long do drawdowns take to recover? One artifact debunked, one honest
null, one robust finding, one deliberately-sprung statistical trap.

## Why survival analysis at all

The naive approach to "how long until X" — average the durations you observed — is
biased the moment some episodes haven't finished: you either drop them (understating
long durations, since the longest episodes are exactly the unfinished ones) or count
them as complete (same bias). **Right-censoring** handles it: an unfinished episode
contributes "lasted at least this long". Kaplan-Meier is the standard estimator that
uses censored observations correctly: they stay in the risk set until they leave, they
just never contribute an event.

## The estimators, from scratch

- **Kaplan-Meier** (`kaplan_meier.py`): sort durations; at each distinct event time
  t_i with n_i still at risk and d_i events, multiply survival by (1 − d_i/n_i).
  Variance via Greenwood's formula. ~60 lines. Tests include a fully hand-worked
  5-observation example and exact cross-validation against `lifelines` on random
  censored data — the same "independent implementations must agree" discipline as the
  options library.
- **Log-rank test** (`logrank.py`): at each event time, the events are a
  hypergeometric draw from the pooled risk set under the null of equal hazards;
  sum observed-minus-expected for one group, standardise, compare to chi²(1).
- **Cox proportional hazards**: via `lifelines` (implementing the partial likelihood
  well is a project of its own). Models the *hazard* — the instantaneous recovery
  rate — as baseline × exp(β·covariates). exp(β) is a hazard ratio: above 1 speeds
  recovery, below 1 slows it.

## The episode machinery

`drawdowns.py` walks a price series once: an episode opens on a close 10% below the
trailing 252-day high, deepens freely, and closes when the entry-time high is
regained; sample-end closes it as censored. Baseline covariates are frozen at entry:
trailing 60-day realised vol, and SPY's own drawdown that day. The eventual `max_depth`
is also recorded — *deliberately*, as the bait for the trap exhibit. Tests construct
tiny price paths where every episode boundary, duration, and depth is known by hand.

## What the results actually showed

1. **The scaling artifact (the study's best lesson).** Raw: semi 10% drawdowns
   recover in median 34 trading days vs SPY's 81 — "semis recover faster!" But 10%
   is ~a quarter of an annual sigma for a 45%-vol semi and over half a sigma for
   19%-vol SPY: different severities entirely. Severity-matched (semis at 20%), semi
   median goes to 116 days and the log-rank test collapses to p ≈ 0.87. The effect
   was an artifact of the threshold, and the study catches its own artifact —
   which is the behaviour that makes research trustworthy.
2. **The honest split on covariates.** With ticker-clustered errors (episodes across
   names share market shocks — 2008 hits all five, so 162 episodes ≠ 162 independent
   observations): trailing vol at entry is null (p ≈ 0.23); concurrent market
   drawdown is robustly negative — hazard ratio ~0.06 per unit, i.e. ~24% lower
   recovery hazard per 10 points of SPY drawdown (exp(−2.77 × 0.1) ≈ 0.76,
   p ≈ 1e-4). Reading: you can't regain your high while the market that dragged you
   down is still falling. Idiosyncratic dips heal fast; systematic ones wait.
3. **The trap.** Add `max_depth` as a covariate and the model turns spectacular —
   most instructively, *trailing vol flips from null to hazard ratio ≈ 54,
   p < 1e-4*. Why it's fake: depth is measured over the episode's future (you can't
   know it at entry), and mechanically, longer episodes have more time to deepen and
   deeper troughs have further to climb — so depth is partly the outcome itself,
   dressed as a predictor. Conditioning on it distorts every other coefficient
   (among episodes forced to equal final depth, high-vol names do bounce faster —
   a comparison unavailable to any forecaster). This is the survival-analysis
   sibling of lookahead bias in backtesting: repo 2 blocks it with an execution
   lag, here it's blocked by the baseline-covariates-only rule.

## The design-win pipeline

`designwins.py` fixes the corpus schema (ticker, announce_date, customer, segment,
source_url, optional revenue_event_date), validates loudly (unknown segments, revenue
before announcement, missing columns all raise), and `to_survival_table` converts a
corpus into (duration, event) rows — wins without an observed revenue event are
censored at the `as_of` date. The template in `data/` contains two clearly-marked
EXAMPLE rows showing the format. Hand-collecting ~50 real announcements is the next
step; the analysis then runs unchanged.

## Likely interview questions

- *Why not just average the recovery times?* Censoring: the unfinished episodes are
  disproportionately the longest ones. Dropping or truncating them biases the mean
  down. KM is the estimator that uses exactly the information each observation has.
- *Why cluster by ticker?* Episodes overlap in calendar time — a market crash opens
  episodes in all five names that then share their recovery clock. Treating them as
  independent overstates the effective sample and understates standard errors.
- *What makes a covariate "post-baseline" and why does it break Cox?* Anything not
  knowable at the episode's start. The hazard model asks "given what was known at
  entry, what's the recovery rate?" — conditioning on future information answers a
  different, non-causal, non-forecastable question while wearing the same notation.
- *How would the design-win study differ?* Same machinery; the event becomes
  "revenue acceleration attributed to the win", covariates become announcement-date
  facts (segment, customer type, cycle state), and the main statistical risks become
  announcement selection bias (companies announce the wins they're proud of) and
  event-attribution noise in quarterly revenue.
