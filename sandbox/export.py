# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get the whole run out as a table, and stop needing JAX.

The fairness and visualisation pathways do not require anybody to learn this
codebase. Run one rollout, call :func:`to_dataframe`, and the rest is pandas,
Observable, R, or a spreadsheet.

Two frames, because the run has two natural grains:

* :func:`to_dataframe` -- one row per interval per connection point. Long
  format, so a group-by answers most questions in one line.
* :func:`feeder_dataframe` -- one row per interval, for the network as a whole.

The connection frame carries **all eighteen** points, tenants included. That
matters more than it sounds: six households have no inverter and therefore no
agent, so they are missing entirely from anything indexed by agent -- and they
are precisely the households a fairness audit exists to ask about.
"""

from typing import Optional

import numpy as np
import pandas as pd

from sandbox.rollout import Trajectory
from sandbox.scenarios import STEPS_PER_DAY, Population, step_duration_h


def to_dataframe(
    trajectory: Trajectory,
    population: Population,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """One row per interval per connection point.

    Columns:
        ``interval`` -- step index from the start of the episode.
        ``day`` / ``day_step`` / ``hour`` -- clock, for time-of-day grouping.
        ``connection`` -- connection point index, 0 to 17.
        ``household`` -- its type: tenant, pv_only, pv_battery, large_flex.
        ``distance_rank`` -- 0 nearest the transformer, 17 furthest.
        ``has_inverter`` -- False for the six tenants.
        ``meter_kw`` -- net power, positive when injecting.
        ``settlement_chf`` -- what it earned (positive) or paid (negative).
        ``voltage_pu`` -- voltage at its own bus.
        ``p_set_kw`` / ``pv_available_kw`` / ``pv_realized_kw`` -- agents only,
        NaN at a tenant's row, since a household with no inverter sets nothing.
        ``run`` -- the `label`, so several runs concatenate cleanly.
    """
    n_steps, num_pq = np.asarray(trajectory.meter_kwh).shape
    interval = np.repeat(np.arange(n_steps), num_pq)
    connection = np.tile(np.arange(num_pq), n_steps)
    day_step = np.repeat(np.asarray(trajectory.day_step), num_pq)

    inverter_of_pq = {pq: agent for agent, pq in enumerate(population.inverter_id)}
    agent_index = np.array([inverter_of_pq.get(pq, -1) for pq in range(num_pq)])
    has_inverter = agent_index >= 0

    def per_agent(values: np.ndarray) -> np.ndarray:
        """Scatter an agent-indexed array onto connection points, NaN elsewhere."""
        wide = np.full((n_steps, num_pq), np.nan, dtype=np.float64)
        wide[:, has_inverter] = np.asarray(values)[:, agent_index[has_inverter]]
        return wide.reshape(-1)

    voltage = np.asarray(trajectory.voltage_pu)
    # voltage_pu is indexed by BUS; connection points are a subset.
    bus_of_pq = np.asarray(population.pq_bus_id())

    frame = pd.DataFrame(
        {
            "interval": interval,
            "day": interval // STEPS_PER_DAY,
            "day_step": day_step,
            "hour": day_step * (24.0 / STEPS_PER_DAY),
            "connection": connection,
            "household": [population.type_of_pq[pq] for pq in connection],
            "distance_rank": [population.distance_rank[pq] for pq in connection],
            "has_inverter": np.tile(has_inverter, n_steps),
            "meter_kw": (np.asarray(trajectory.meter_kwh) / step_duration_h()).reshape(-1),
            "settlement_chf": np.asarray(trajectory.settlement_chf).reshape(-1),
            "voltage_pu": voltage[:, bus_of_pq].reshape(-1),
            "p_set_kw": per_agent(trajectory.p_set_kw),
            "pv_available_kw": per_agent(trajectory.pv_available_kw),
            "pv_realized_kw": per_agent(trajectory.pv_realized_kw),
        }
    )
    if label is not None:
        frame["run"] = label
    return frame


def feeder_dataframe(
    trajectory: Trajectory,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """One row per interval, for the network as a whole.

    ``transformer_kw`` follows the load convention: positive when the feeder
    draws from the grid, negative when it exports into it.
    """
    n_steps = int(np.asarray(trajectory.transformer_kw).shape[0])
    day_step = np.asarray(trajectory.day_step)
    frame = pd.DataFrame(
        {
            "interval": np.arange(n_steps),
            "day": np.arange(n_steps) // STEPS_PER_DAY,
            "day_step": day_step,
            "hour": day_step * (24.0 / STEPS_PER_DAY),
            "transformer_kw": np.asarray(trajectory.transformer_kw),
            "transformer_kvar": np.asarray(trajectory.transformer_kvar),
            "losses_kw": np.asarray(trajectory.losses_kw),
            "voltage_max_pu": np.asarray(trajectory.voltage_pu).max(axis=1),
            "voltage_min_pu": np.asarray(trajectory.voltage_pu).min(axis=1),
            "curtailed_kw": np.maximum(
                np.asarray(trajectory.pv_available_kw) - np.asarray(trajectory.pv_realized_kw),
                0.0,
            ).sum(axis=1),
        }
    )
    if label is not None:
        frame["run"] = label
    return frame


def write(
    trajectory: Trajectory,
    population: Population,
    directory: str,
    label: str = "run",
) -> list[str]:
    """Write both frames as Parquet, and return the paths.

    Parquet rather than CSV because the connection frame is twelve thousand
    rows for a single week and every tool that matters reads it.
    """
    from pathlib import Path

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in (
        ("connections", to_dataframe(trajectory, population, label)),
        ("feeder", feeder_dataframe(trajectory, label)),
    ):
        path = target / f"{label}_{name}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(str(path))
    return paths
