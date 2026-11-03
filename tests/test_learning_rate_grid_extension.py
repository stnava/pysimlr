"""Unit tests for learning rate probe grid extension under Rule R11."""
from pysimlr.optimizers import LR_PROBE_GRID, extend_probe_grid


def test_probe_grid_default_is_strictly_decreasing():
    """Default probe grid must be strictly decreasing."""
    assert all(x > y for x, y in zip(LR_PROBE_GRID[:-1], LR_PROBE_GRID[1:]))
    assert LR_PROBE_GRID[0] == 1.0
    assert LR_PROBE_GRID[-1] == 0.001


def test_extend_probe_grid_upper_one_step():
    """Upper 1 step extends to 3.0, strictly decreasing."""
    grid = extend_probe_grid(upper_steps=1)
    assert grid[0] == 3.0
    assert 3.0 in grid
    assert all(x > y for x, y in zip(grid[:-1], grid[1:]))
    assert grid[1:] == LR_PROBE_GRID


def test_extend_probe_grid_upper_two_steps():
    """Upper 2 steps extends to 10.0 and 3.0, strictly decreasing."""
    grid = extend_probe_grid(upper_steps=2)
    assert grid[:2] == (10.0, 3.0)
    assert 10.0 in grid and 3.0 in grid
    assert all(x > y for x, y in zip(grid[:-1], grid[1:]))
    assert grid[2:] == LR_PROBE_GRID


def test_extend_probe_grid_lower_one_step():
    """Lower 1 step extends to 0.0003, strictly decreasing."""
    grid = extend_probe_grid(lower_steps=1)
    assert grid[-1] == 0.0003
    assert 0.0003 in grid
    assert all(x > y for x, y in zip(grid[:-1], grid[1:]))
    assert grid[:-1] == LR_PROBE_GRID


def test_extend_probe_grid_lower_two_steps():
    """Lower 2 steps extends to 0.0003 and 0.0001, strictly decreasing."""
    grid = extend_probe_grid(lower_steps=2)
    assert grid[-2:] == (0.0003, 0.0001)
    assert 0.0003 in grid and 0.0001 in grid
    assert all(x > y for x, y in zip(grid[:-1], grid[1:]))
    assert grid[:-2] == LR_PROBE_GRID


def test_extend_probe_grid_both_directions():
    """Simultaneous upper and lower extension is strictly decreasing and contains endpoints."""
    grid = extend_probe_grid(upper_steps=2, lower_steps=2)
    assert grid[0] == 10.0
    assert grid[-1] == 0.0001
    assert 3.0 in grid and 0.0003 in grid
    assert all(x > y for x, y in zip(grid[:-1], grid[1:]))
