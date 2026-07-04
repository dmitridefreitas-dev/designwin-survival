"""survivalsemi — survival analysis for semiconductor equity questions.

Modules:
    kaplan_meier  KM estimator with Greenwood CIs, implemented from scratch
    logrank       two-sample log-rank test, implemented from scratch
    drawdowns     price series -> drawdown-recovery episodes (right-censored)
    designwins    schema + loader for a hand-collected design-win corpus
    data          cached daily price downloads

The from-scratch estimators are cross-validated against lifelines in tests;
Cox regression uses lifelines directly (see the study notebook).
"""

from survivalsemi.kaplan_meier import KaplanMeier

__all__ = ["KaplanMeier"]
