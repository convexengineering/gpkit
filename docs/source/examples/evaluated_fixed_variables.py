# code from t_GPSubs.test_calcconst in tests/t_sub.py
x = Variable("x", "hours")
t_day = Variable("t_{day}", 12, "hours")
t_night = Variable("t_{night}", lambda c: 24 - c[t_day], "hours")
# note that t_night has a function as its value
m = Model(x, [x >= t_day, x >= t_night])
sol = m.solve(verbosity=0)
self.assertAlmostEqual(sol(t_night)/gpkit.ureg.hours, 12)
m.substitutions.update({t_day: ("sweep", [8, 12, 16])})
sol = m.solve(verbosity=0)
self.assertEqual(len(sol["cost"]), 3)
npt.assert_allclose(sol(t_day) + sol(t_night), 24)