"""The optimizer benchmark's generator must keep the properties it relies on:
a wrong PCA start, a recoverable shared signal, and a normalised gap."""
import importlib.util
import pathlib

import numpy as np

spec = importlib.util.spec_from_file_location(
    "optbench", pathlib.Path(__file__).parents[1] / "scripts" / "optimizer_benchmark_synthetic.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


def test_start_is_wrong_and_signal_is_recoverable():
    from sklearn.cross_decomposition import CCA
    from pysimlr.simlr import simlr
    z, views, sups = B.planted_buried(0)
    x = [((v - v.mean(0)) / v.std(0)).numpy() for v in views]
    c0, c1 = CCA(n_components=2, max_iter=2000).fit(x[0], x[1]).transform(x[0], x[1])
    assert B.latent_r2(z, (c0 + c1) / 2) > 0.9
    init = simlr(views, k=2, iterations=0, energy_type="acc", positivity="positive", random_state=0)
    assert B.latent_r2(z, init["u"]) < 0.4
    assert all(len(s) == 2 for s in sups)


def test_support_f1_is_one_on_the_truth():
    v = np.zeros((10, 2)); v[[0, 1, 2], 0] = 1; v[[5, 6], 1] = 1
    assert B.support_f1(v[:, ::-1], [{0, 1, 2}, {5, 6}]) == 1.0
