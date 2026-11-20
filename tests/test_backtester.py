"""
Backtester tests. Run from the project folder with:

    python3 -m unittest discover -s tests -v

They use synthetic prices (seeded random walk), so they need no network and no API key.
"""

import os
import sys
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import strategy_dsl as dsl  # noqa: E402
import analysis             # noqa: E402


def make_prices(n=1500, seed=7, drift=0.0003, vol=0.015, start=100.0):
    """Daily OHLCV with realistic structure: gaps at the open, high >= max(open, close)."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    prev = np.concatenate([[start], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


SPECS = {
    "golden_cross": {"name": "golden", "direction": "long",
                     "indicators": [{"id": "a", "type": "sma", "window": 50},
                                    {"id": "b", "type": "sma", "window": 200}],
                     "entry": {"left": "a", "op": "crosses_above", "right": "b"},
                     "exit": {"left": "a", "op": "crosses_below", "right": "b"},
                     "risk": {"stop_loss_pct": 0.08}},
    "breakout_trailing": {"name": "breakout", "direction": "long",
                          "indicators": [{"id": "hh", "type": "highest", "window": 20, "offset": 1},
                                         {"id": "ll", "type": "lowest", "window": 10, "offset": 1}],
                          "entry": {"left": "close", "op": ">", "right": "hh"},
                          "exit": {"left": "close", "op": "<", "right": "ll"},
                          "risk": {"trailing_stop_pct": 0.10}},
    "zscore_dip": {"name": "z", "direction": "long",
                   "indicators": [{"id": "z", "type": "zscore", "window": 20}],
                   "entry": {"left": "z", "op": "<", "right": -2},
                   "exit": {"left": "z", "op": ">", "right": 0},
                   "risk": {"stop_loss_pct": 0.07, "max_hold_days": 15}},
    "short_bbands": {"name": "short bb", "direction": "short",
                     "indicators": [{"id": "u", "type": "bbands", "window": 20, "field": "upper"},
                                    {"id": "m", "type": "bbands", "window": 20, "field": "middle"}],
                     "entry": {"left": "close", "op": ">", "right": "u"},
                     "exit": {"left": "close", "op": "<", "right": "m"},
                     "risk": {"stop_loss_pct": 0.05, "take_profit_pct": 0.05},
                     "position_size": 0.5},
}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_prices()
        cls.specs = {k: dsl.validate_spec(v) for k, v in SPECS.items()}

    def test_no_lookahead(self):
        """Running on data cut at bar k must reproduce the full run up to bar k exactly.
        If any rule peeked at future bars, truncation would change earlier decisions."""
        for name, spec in self.specs.items():
            full = dsl.run_spec(self.df, spec).df
            for cut in (300, 700, 1200):
                part = dsl.run_spec(self.df.iloc[:cut], spec).df
                with self.subTest(spec=name, cut=cut):
                    np.testing.assert_allclose(part["PortfolioValue"], full["PortfolioValue"].iloc[:cut])
                    self.assertTrue((part["Position"] == full["Position"].iloc[:cut]).all())

    def test_accounting_reconciles_from_fills(self):
        """Replaying every trade's entry/exit fills with costs must land on the final equity."""
        for name, spec in self.specs.items():
            res = dsl.run_spec(self.df, spec)
            c, size = spec["costs_bps"] / 1e4, spec["position_size"]
            sign = 1 if spec["direction"] == "long" else -1
            eq = 10_000.0
            for t in res.trades:
                n = sign * eq * size / t["entry_price"]
                cash = eq - n * t["entry_price"] - abs(n * t["entry_price"]) * c
                if t["exit_date"]:
                    eq = cash + n * t["exit_price"] - abs(n * t["exit_price"]) * c
                else:   # still open: equity is marked at the last close, no exit cost yet
                    eq = cash + n * self.df["Close"].iloc[-1]
            with self.subTest(spec=name):
                self.assertAlmostEqual(eq, res.df["PortfolioValue"].iloc[-1], places=6)

    def test_trade_returns_are_net_of_costs(self):
        """With 100% sizing, compounding the per-trade returns must equal the equity curve,
        which only works if those returns include both sides' costs."""
        for direction in ("long", "short"):
            spec = dict(SPECS["breakout_trailing"], direction=direction, costs_bps=25)
            if direction == "short":
                spec = dict(spec, entry={"left": "close", "op": "<", "right": "ll"},
                            exit={"left": "close", "op": ">", "right": "hh"})
            spec = dsl.validate_spec(spec)
            res = dsl.run_spec(self.df, spec)
            closed = [t for t in res.trades if t["exit_date"]]
            if res.trades and not res.trades[-1]["exit_date"]:
                res = dsl.run_spec(self.df.loc[:res.trades[-1]["entry_date"]].iloc[:-1], spec)
                closed = [t for t in res.trades if t["exit_date"]]
            with self.subTest(direction=direction):
                self.assertGreater(len(closed), 5)
                compounded = 10_000 * np.prod([1 + t["return"] for t in closed])
                self.assertAlmostEqual(compounded, res.df["PortfolioValue"].iloc[-1], places=4)
                for t in closed:
                    self.assertLess(t["return"], t["gross_return"])

    def test_zero_costs_net_equals_gross(self):
        spec = dsl.validate_spec(dict(SPECS["golden_cross"], costs_bps=0))
        for t in dsl.run_spec(self.df, spec).trades:
            self.assertAlmostEqual(t["return"], t["gross_return"])

    def test_stop_loss_caps_loss_unless_gapped(self):
        spec = self.specs["zscore_dip"]
        for t in dsl.run_spec(self.df, spec).trades:
            if t["exit_reason"] == "stop loss":
                stop = t["entry_price"] * (1 - 0.07)
                self.assertLessEqual(t["exit_price"], stop + 1e-9)   # filled at stop, or worse on a gap


class PerturbTests(unittest.TestCase):
    def _value(self, spec, path):
        node = spec
        for p in path:
            node = node[p]
        return node

    def test_negative_threshold_keeps_sign(self):
        spec = dsl.validate_spec(SPECS["zscore_dip"])
        path = next(p for p, v, label in dsl.numeric_params(spec) if v == -2)
        self.assertEqual(self._value(dsl.perturb(spec, path, 0.75), path), -1.5)
        self.assertEqual(self._value(dsl.perturb(spec, path, 1.25), path), -2.5)

    def test_small_threshold_not_clamped(self):
        spec = dsl.validate_spec({"direction": "long", "indicators": [{"id": "z", "type": "zscore", "window": 20}],
                                  "entry": {"left": "z", "op": ">", "right": 1},
                                  "exit": {"left": "z", "op": "<", "right": 0}})
        path = dsl.numeric_params(spec)[0][0]
        self.assertEqual(self._value(dsl.perturb(spec, path, 0.75), path), 0.75)
        self.assertEqual(self._value(dsl.perturb(spec, path, 1.25), path), 1.25)

    def test_windows_stay_whole_bars(self):
        spec = dsl.validate_spec(SPECS["zscore_dip"])
        path = ("indicators", 0, "window")
        self.assertEqual(self._value(dsl.perturb(spec, path, 0.75), path), 15)
        self.assertIsInstance(self._value(dsl.perturb(spec, path, 1.3), path), int)

    def test_every_variant_still_validates(self):
        for name, raw in SPECS.items():
            spec = dsl.validate_spec(raw)
            for path, _, _ in dsl.numeric_params(spec):
                for f in (0.75, 1.25):
                    with self.subTest(spec=name, path=path, factor=f):
                        dsl.validate_spec(dsl.perturb(spec, path, f))


class ScorecardTests(unittest.TestCase):
    DIAG = {"split": {"split_date": "2021-01-01", "in_sample": {"sharpe": 0.1}, "out_of_sample": {"sharpe": 0.2}},
            "costs": [{"bps": 25, "sharpe": 0.5}], "sensitivity": []}

    def _first(self, **m):
        base = {"sharpe_ratio": 0.5, "bah_sharpe": 0.8, "total_return": 0.2, "n_round_trips": 40}
        base.update(m)
        return analysis.checks(base, self.DIAG)[0]

    def test_losing_strategy_fails(self):
        self.assertEqual(self._first(total_return=-0.43, sharpe_ratio=-0.9), {"status": "fail", "label": "Loses money"})

    def test_beats_buy_and_hold(self):
        self.assertEqual(self._first(sharpe_ratio=0.9)["status"], "pass")

    def test_close_and_trailing(self):
        self.assertEqual(self._first(sharpe_ratio=0.7)["status"], "warn")
        self.assertEqual(self._first(sharpe_ratio=0.3)["status"], "fail")

    def test_full_pipeline_on_a_loser(self):
        df = make_prices(drift=0.0008)          # strong uptrend: shorting it should lose
        spec = dsl.validate_spec(SPECS["short_bbands"])
        res = dsl.run_spec(df, spec)
        m = analysis.compute_metrics(res)
        diag = analysis.diagnostics(df, spec, res, m)
        self.assertLess(m["total_return"], 0)
        self.assertIn("fail", [c["status"] for c in diag["checks"]])
        self.assertTrue(any("Loses money" in f for f in diag["red_flags"]))


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_specs(self):
        bad = [
            {"entry": {}},                                                      # no entry
            {"indicators": [{"id": "s", "type": "sma", "window": 0}],
             "entry": {"left": "close", "op": ">", "right": "s"}, "exit": {"left": "close", "op": "<", "right": "s"}},
            {"indicators": [], "entry": {"left": "close", "op": ">", "right": "nope"}, "risk": {"stop_loss_pct": 0.05}},
            {"indicators": [], "entry": {"left": "close", "op": ">", "right": 1}},  # no way out
            {"indicators": [], "entry": {"left": "close", "op": ">", "right": 1}, "risk": {"stop_loss_pct": 5}},
        ]
        for spec in bad:
            with self.subTest(spec=spec):
                with self.assertRaises(dsl.SpecError):
                    dsl.validate_spec(spec)


class DataCacheTests(unittest.TestCase):
    """engine.load_data: one download per ticker per day, sliced locally, end date exclusive."""

    def setUp(self):
        import engine
        self.engine = engine
        engine.clear_cache()
        self.prices = make_prices(n=600)

    def tearDown(self):
        self.engine.clear_cache()

    def test_downloads_once_then_slices(self):
        with mock.patch.object(self.engine.yf, "download", return_value=self.prices) as dl:
            a = self.engine.load_data("spy", "2015-03-02", "2016-01-04")
            b = self.engine.load_data("SPY", "2015-06-01", "2016-06-01")
        self.assertEqual(dl.call_count, 1)                      # second call served from cache
        self.assertEqual(a.index[0], pd.Timestamp("2015-03-02"))
        self.assertLess(a.index[-1], pd.Timestamp("2016-01-04"))  # end is exclusive, like yfinance
        self.assertEqual(b.index[0], pd.Timestamp("2015-06-01"))

    def test_refreshes_on_a_new_day(self):
        with mock.patch.object(self.engine.yf, "download", return_value=self.prices) as dl:
            self.engine.load_data("SPY", "2015-03-02", "2016-01-04")
            stale = self.engine._CACHE["SPY"][1]
            self.engine._CACHE["SPY"] = (self.engine.date(2000, 1, 1), stale)
            self.engine.load_data("SPY", "2015-03-02", "2016-01-04")
        self.assertEqual(dl.call_count, 2)

    def test_bad_ticker_raises_and_is_not_cached(self):
        with mock.patch.object(self.engine.yf, "download", return_value=pd.DataFrame()):
            with self.assertRaises(ValueError):
                self.engine.load_data("NOPE", "2015-01-01", "2020-01-01")
        self.assertNotIn("NOPE", self.engine._CACHE)


try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False


@unittest.skipUnless(HAVE_FLASK, "flask not installed")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as app_module
        cls.app_module = app_module
        cls.prices = make_prices(n=2500)

    def setUp(self):
        fake = lambda t, s, e: self.prices.loc[s:e].copy()
        self.patch = mock.patch.object(self.app_module.engine, "load_data", side_effect=fake)
        self.patch.start()
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.patch.stop()

    def test_default_end_date_is_today(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn(datetime.now().strftime("%Y-%m-%d"), html)

    def test_manual_mode_matches_ai_engine(self):
        """The Manual MA form and the same rules sent as a spec to /analyze must agree."""
        form = {"ticker": "SPY", "strategy": "MA Crossover", "ma_window": "50",
                "start_date": "2015-01-01", "end_date": "2024-01-01"}
        html = self.client.post("/", data=form).get_data(as_text=True)
        spec, _ = self.app_module.manual_spec({"strategy": "MA Crossover", "ma_window": 50})
        j = self.client.post("/analyze", json={"spec": spec, "review": False, "ticker": "SPY",
                                               "start_date": "2015-01-01", "end_date": "2024-01-01"}).get_json()
        self.assertEqual(j["status"], "ok")
        self.assertIn(f"${j['metrics']['final_value']:.2f}", html)

    def test_manual_rsi_runs(self):
        form = {"ticker": "SPY", "strategy": "RSI", "rsi_window": "14", "oversold": "30", "overbought": "70",
                "start_date": "2015-01-01", "end_date": "2024-01-01"}
        r = self.client.post("/", data=form)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Final Value", r.get_data(as_text=True))

    def test_manual_rejects_bad_window(self):
        form = {"ticker": "SPY", "strategy": "MA Crossover", "ma_window": "0",
                "start_date": "2015-01-01", "end_date": "2024-01-01"}
        self.assertIn("window must be an integer", self.client.post("/", data=form).get_data(as_text=True))

    def test_bad_dates_are_400_not_500(self):
        r = self.client.post("/analyze", json={"spec": SPECS["golden_cross"], "review": False,
                                               "start_date": "garbage"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/analyze", json={"spec": SPECS["golden_cross"], "review": False,
                                               "start_date": "2024-01-01", "end_date": "2020-01-01"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
