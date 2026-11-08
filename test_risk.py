import unittest
import numpy as np
from portfolio_risk import simulate, risk, stress_correlation


class RiskTests(unittest.TestCase):
    def test_known_tail(self):
        result = risk(np.arange(1, 101), 100)
        self.assertAlmostEqual(result['VaR_95_dollars'], 94.05)
        self.assertAlmostEqual(result['ES_95_dollars'], 97)

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
