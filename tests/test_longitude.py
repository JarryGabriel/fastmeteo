"""Longitudes around Greenwich, for global (0..360) and regional (-32..42) grids.

Offline: synthetic datasets with random values per grid column, so the only
correct answer is the linear interpolation between the neighbouring columns.
"""

from __future__ import annotations

import tempfile
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fastmeteo.core.grid import Grid
from fastmeteo.source import Arpege

HOUR = pd.Timestamp("2024-06-01T01:00")
LAT = np.array([48.0, 47.0, 46.0])  # descending, like the real sources
RNG = np.random.default_rng(0)


def _field(lon: np.ndarray) -> np.ndarray:
    return RNG.uniform(200.0, 250.0, size=(1, LAT.size, lon.size))


class StubGrid(Grid):
    """Grid returning one hour of a synthetic temperature field."""

    features: ClassVar[list[str]] = ["temperature"]

    def __init__(self, local_store: str, lon: np.ndarray) -> None:
        self.local_store = local_store
        self.lon = lon
        self.values = _field(lon)

    def select_remote(self, hour: pd.DatetimeIndex) -> xr.Dataset:
        ds = xr.Dataset(
            {"temperature": (["time", "latitude", "longitude"], self.values)},
            coords={
                "time": pd.DatetimeIndex([hour]),
                "latitude": LAT,
                "longitude": self.lon,
            },
        )
        ds.time.encoding["units"] = "hours since 2024-01-01"
        ds.time.encoding["dtype"] = "int64"
        return ds

    def coords(self, flight: pd.DataFrame) -> dict[str, Any]:
        return {
            "time": ("points", pd.to_datetime(flight.timestamp).values),
            "latitude": ("points", flight.latitude.values),
            "longitude": ("points", flight.longitude_360.values),
        }

    def column(self, lon: float) -> float:
        return float(self.values[0, 1, int(np.argmin(np.abs(self.lon - lon)))])


def _flight(lons: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": HOUR, "latitude": 47.0, "longitude": lons, "altitude": 35_000.0}
    )


def test_to_grid_frame_follows_the_dataset_convention() -> None:
    from fastmeteo.core.grid import to_grid_frame

    np.testing.assert_allclose(
        to_grid_frame([-0.1, 359.9, 1.0], np.arange(0, 360)), [359.9, 359.9, 1.0]
    )
    np.testing.assert_allclose(
        to_grid_frame([-3.0, 357.0, 10.0], np.arange(-32, 43)), [-3.0, -3.0, 10.0]
    )


def test_close_longitude_only_extends_global_grids() -> None:
    from fastmeteo.core.grid import close_longitude

    glob = StubGrid("unused", np.arange(0.0, 360.0, 1.0)).select_remote(HOUR)
    closed = close_longitude(glob)
    assert closed.longitude.values[-1] == 360.0
    np.testing.assert_array_equal(
        closed.temperature.sel(longitude=360.0), glob.temperature.sel(longitude=0.0)
    )
    regional = StubGrid("unused", np.arange(-32.0, 43.0, 1.0)).select_remote(HOUR)
    assert close_longitude(regional).longitude.size == regional.longitude.size


def test_global_grid_interpolates_across_the_seam() -> None:
    with tempfile.TemporaryDirectory() as store:
        grid = StubGrid(store, np.arange(0.0, 360.0, 1.0))
        out = grid.interpolate(_flight([-0.5, 359.25, 0.5]))
    last, first, second = grid.column(359.0), grid.column(0.0), grid.column(1.0)
    expected = [
        0.5 * last + 0.5 * first,
        0.75 * last + 0.25 * first,
        0.5 * first + 0.5 * second,
    ]
    np.testing.assert_allclose(out.temperature, expected)


def test_regional_grid_west_of_greenwich_reads_the_right_columns() -> None:
    with tempfile.TemporaryDirectory() as store:
        grid = StubGrid(store, np.arange(-32.0, 43.0, 1.0))
        out = grid.interpolate(_flight([-3.0, -0.5, 3.0]))
    expected = [
        grid.column(-3.0),
        0.5 * (grid.column(-1.0) + grid.column(0.0)),
        grid.column(3.0),
    ]
    np.testing.assert_allclose(out.temperature, expected)


def test_arpege_local_interpolate_west_of_greenwich(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lon = np.arange(-32.0, 42.5, 0.5)
    t = RNG.uniform(210.0, 230.0, size=(1, 2, LAT.size, lon.size))
    ds = xr.Dataset(
        {
            name: (["time", "isobaricInhPa", "latitude", "longitude"], t)
            for name in ("u", "v", "t")
        },
        coords={
            "time": [HOUR],
            "isobaricInhPa": [200, 300],
            "latitude": LAT,
            "longitude": lon,
        },
    )
    grid = Arpege(local_store="unused", features=["t"], resolution="01")
    monkeypatch.setattr(grid, "select_remote", lambda hour: ds)
    flight = _flight([-3.0, 3.0]).assign(altitude=38_662.0)  # ~ 200 hPa
    out = grid.local_interpolate(flight)
    expected = ds.t.sel(
        time=HOUR, isobaricInhPa=200, latitude=47.0, longitude=[-3.0, 3.0]
    ).values
    np.testing.assert_allclose(out.t, expected, rtol=0, atol=0.05)
