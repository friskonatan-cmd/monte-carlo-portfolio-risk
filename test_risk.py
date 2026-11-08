import unittest
from statistics import NormalDist
import numpy as np
from portfolio_risk import (simulate, simulate_terminal, historical_terminal,
                            risk, stress_correlation, monte_carlo_intervals)


class RiskTests(unittest.TestCase):
    def test_known_tail(self):
        result = risk(np.arange(1, 101), 100)
        self.assertAlmostEqual(result['VaR_dollars'], 94)
        self.assertAlmostEqual(result['ES_dollars'], 97)

    def test_ties_do_not_dilute_expected_shortfall(self):
        # Worst 5% = 4% of losses at 100 plus 1% of losses at zero.
        losses = np.array([0]*96 + [100]*4)
        result = risk(100-losses, 100)
        self.assertEqual(result['VaR_dollars'], 0)
        self.assertAlmostEqual(result['ES_dollars'], 80)

    def test_fractional_tail_and_confidence(self):
        result = risk(100-np.array([0, 10, 20, 30]), 100, .625)
        self.assertAlmostEqual(result['ES_dollars'], (30+.5*20)/1.5)
        self.assertEqual(result['VaR_dollars'], 20)
        result99 = risk(np.arange(1, 101), 100, .99)
        self.assertEqual(result99['VaR_dollars'], 98)
        self.assertAlmostEqual(result99['ES_dollars'], 99)
        # Gains must remain negative losses, not be clipped to zero.
        self.assertEqual(risk([110]*10, 100)['ES_dollars'], -10)

    def test_invalid_risk_inputs(self):
        for values in [[], [np.nan], [np.inf], [[100, 110]]]:
            with self.assertRaises(ValueError):
                risk(values, 100)
        for confidence in [0, 1, np.nan]:
            with self.assertRaises(ValueError):
                risk([100], 100, confidence)

    def test_terminal_shortcut_matches_same_daily_shocks(self):
        z = np.random.default_rng(13).standard_normal((1000, 21, 2))
        mu, sigma, w = np.array([.001, -.0002]), np.array([.02, .01]), np.array([.4, .6])
        c = np.array([[1, .6], [.6, 1]])
        full = simulate(mu, sigma, c, w, z, 10000)[:, -1]
        fast = simulate_terminal(mu, sigma, c, w, z.sum(axis=1)/np.sqrt(21), 10000, 21)
        np.testing.assert_allclose(full, fast, rtol=1e-13)

    def test_single_asset_matches_analytic_lognormal_tail(self):
        # Independent analytic benchmark for the sampler and risk estimators.
        mu, sigma, days, initial, confidence = .0003, .02, 21, 10000, .95
        z = np.random.default_rng(9).standard_normal((300000, 1))
        values = simulate_terminal(np.array([mu]), np.array([sigma]), np.eye(1),
                                   np.ones(1), z, initial, days)
        norm = NormalDist()
        q = norm.inv_cdf(1-confidence)
        a, b = days*mu, np.sqrt(days)*sigma
        exact_var = initial*(1-np.exp(a+b*q))
        exact_es = initial*(1-np.exp(a+b*b/2)*norm.cdf(q-b)/(1-confidence))
        result = risk(values, initial, confidence)
        self.assertAlmostEqual(result['VaR_dollars'], exact_var, delta=15)
        self.assertAlmostEqual(result['ES_dollars'], exact_es, delta=15)

    def test_historical_sampling_preserves_joint_days_and_compounds(self):
        logs = np.log([[1.1, .9], [.9, 1.1]])
        one_day = historical_terminal(logs, np.array([.5, .5]), 100, 1, 100,
                                      np.random.default_rng(4), 1)
        np.testing.assert_allclose(one_day, 100)  # fails if columns sampled separately
        two_days = historical_terminal(logs, np.array([.5, .5]), 100, 2, 100,
                                       np.random.default_rng(4), 2)
        np.testing.assert_allclose(two_days, 99)
        three_days = historical_terminal(logs, np.array([.5, .5]), 100, 3, 100,
                                         np.random.default_rng(4), 2)
        np.testing.assert_allclose(three_days, 99)  # truncated final block, no extra day

    def test_intervals_reproducible_and_scale_in_dollars(self):
        values = np.linspace(80, 120, 1000)
        a = monte_carlo_intervals(values, 100, .95, np.random.default_rng(2), 100)
        b = monte_carlo_intervals(2*values, 200, .95, np.random.default_rng(2), 100)
        for key in a:
            self.assertAlmostEqual(b[key], 2*a[key])
        self.assertLess(a['VaR_mc_low'], a['VaR_mc_high'])

    def test_constant_prices(self):
        v = simulate(np.zeros(2), np.zeros(2), np.eye(2), np.array([.4,.6]),
                     np.zeros((100,21,2)),10000)
        np.testing.assert_allclose(v,10000)

    def test_buy_hold_compounding(self):
        v=simulate(np.log([1.1,.9]),np.zeros(2),np.eye(2),np.array([.5,.5]),np.zeros((1,2,2)),100)
        self.assertAlmostEqual(v[0,-1],101)  # daily rebalancing would give 100

    def test_cholesky_covariance(self):
        c=np.array([[1,.7],[.7,1]])
        z=np.random.default_rng(1).standard_normal((200000,2))
        x=z@np.linalg.cholesky(c).T
        np.testing.assert_allclose(np.cov(x.T),c,atol=.015)

    def test_stress_validity(self):
        c=np.array([[1,-.5],[-.5,1]])
        for strength in [0,.25,.5,.95]:
            s=stress_correlation(c,strength)
            self.assertGreater(np.linalg.eigvalsh(s).min(),0)
            np.testing.assert_allclose(np.diag(s),1)


if __name__=='__main__':
    unittest.main()
