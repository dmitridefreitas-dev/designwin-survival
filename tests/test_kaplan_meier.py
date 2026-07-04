"""KM against a hand-computed example and against lifelines."""

import numpy as np
import pytest
from lifelines import KaplanMeierFitter

from survivalsemi import KaplanMeier


class TestHandComputed:
    """Durations [1, 2e, 3e, 4c, 5e]: events at 2, 3, 5; censored at 1?, 4.

    Worked by hand with durations [1c, 2e, 3e, 4c, 5e]:
      t=2: n=4 at risk, d=1 -> S = 3/4
      t=3: n=3 at risk, d=1 -> S = 3/4 * 2/3 = 1/2
      t=5: n=1 at risk, d=1 -> S = 0
    """

    def fit(self):
        return KaplanMeier.fit([1, 2, 3, 4, 5], [0, 1, 1, 0, 1])

    def test_survival_values(self):
        km = self.fit()
        assert km.event_times.tolist() == [2.0, 3.0, 5.0]
        assert km.survival.tolist() == pytest.approx([0.75, 0.5, 0.0])

    def test_step_lookup(self):
        km = self.fit()
        assert km.survival_at(1.5) == 1.0
        assert km.survival_at(2.0) == 0.75
        assert km.survival_at(4.9) == 0.5
        assert km.survival_at(100.0) == 0.0

    def test_median(self):
        assert self.fit().median() == 3.0

    def test_counts(self):
        km = self.fit()
        assert km.n_observations == 5
        assert km.n_censored == 2


class TestAgainstLifelines:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_survival_curve_matches(self, seed):
        rng = np.random.default_rng(seed)
        n = 400
        durations = rng.exponential(scale=50, size=n).round() + 1
        events = rng.random(n) > 0.3  # ~30% censored

        ours = KaplanMeier.fit(durations, events)
        theirs = KaplanMeierFitter().fit(durations, events)

        for t in ours.event_times:
            assert ours.survival_at(t) == pytest.approx(
                float(theirs.predict(t)), abs=1e-10
            )

    def test_median_matches(self):
        rng = np.random.default_rng(7)
        durations = rng.exponential(scale=30, size=300).round() + 1
        events = rng.random(300) > 0.2

        ours = KaplanMeier.fit(durations, events)
        theirs = KaplanMeierFitter().fit(durations, events)
        assert ours.median() == pytest.approx(float(theirs.median_survival_time_))


class TestValidation:
    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="no observations"):
            KaplanMeier.fit([], [])

    def test_negative_durations_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            KaplanMeier.fit([-1, 2], [1, 1])

    def test_all_censored_curve_stays_at_one(self):
        km = KaplanMeier.fit([5, 10, 15], [0, 0, 0])
        assert len(km.event_times) == 0
        assert km.survival_at(100) == 1.0
        assert np.isnan(km.median())
