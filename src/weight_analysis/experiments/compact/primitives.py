"""Shared float64 block reductions; definitions reuse historical VectorMoments."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch

from ...metrics.centroids import projection_summary
from ...statistics.aggregation import VectorMoments


class TensorStatistics:
    """All compatible pairs, two origins and shared directions from each block once.

    Index zero is the common reference. Direct differences are accumulated
    explicitly instead of subtracting nearly equal Gram diagonals. Centered
    covariance uses the parallel Welford identity. Signs retain the historical
    1e-8 exclusion floor. Only small sufficient statistics cross back to CPU.
    """

    def __init__(self, count, groups, *, device="cpu", state=None):
        self.count, self.groups, self.device, self.n = count, groups, device, 0
        self.data = {}
        for source in ("final", "delta"):
            for key in ("gram", "centered", "sign_n", "sign_agree"):
                self.data[source + "_" + key] = torch.zeros(
                    count, count, dtype=torch.float64, device=device
                )
            self.data[source + "_mean"] = torch.zeros(count, dtype=torch.float64, device=device)
        for key in ("distance", "l1", "maximum"):
            self.data[key] = torch.zeros(count, count, dtype=torch.float64, device=device)
        self.shared = {
            group["group_id"]: torch.zeros(
                len(group["indices"]), 4, dtype=torch.float64, device=device
            )
            for group in groups
        }
        if state:
            self.n = state["n"]
            self.data = {
                key: torch.tensor(value, dtype=torch.float64, device=device)
                for key, value in state["data"].items()
            }
            self.shared = {
                key: torch.tensor(value, dtype=torch.float64, device=device)
                for key, value in state["shared"].items()
            }

    @torch.no_grad()
    def add(self, final):
        final = final.to(device=self.device, dtype=torch.float64)
        if final.ndim != 2 or final.shape[0] != self.count or not final.shape[1]:
            raise ValueError("Expected aligned, nonempty checkpoint blocks")
        if not torch.isfinite(final).all():
            raise ValueError("Nonfinite model parameter")
        delta = final - final[0]
        width = final.shape[1]
        delta_mean = delta.mean(dim=1)
        delta_centered = delta - delta_mean[:, None]
        base_centered = final[0] - final[0].mean()
        gram_delta = delta @ delta.T
        cross = delta @ final[0]
        gram_final = gram_delta + cross[:, None] + cross[None, :] + final[0].square().sum()
        centered_delta = delta_centered @ delta_centered.T
        cross_centered = delta_centered @ base_centered
        centered_final = (
            centered_delta
            + cross_centered[:, None]
            + cross_centered[None, :]
            + base_centered.square().sum()
        )
        for source, values in (("final", final), ("delta", delta)):
            mean = values.mean(dim=1)
            change = mean - self.data[source + "_mean"]
            self.data[source + "_gram"] += gram_final if source == "final" else gram_delta
            self.data[source + "_centered"] += (
                centered_final if source == "final" else centered_delta
            )
            if self.n:
                self.data[source + "_centered"] += torch.outer(change, change) * (
                    self.n * width / (self.n + width)
                )
            self.data[source + "_mean"] += change * (width / (self.n + width))
            for i in range(self.count - 1):
                others = values[i + 1 :]
                mask = (values[i].abs() > 1e-8) & (others.abs() > 1e-8)
                numbers = mask.sum(dim=1).double()
                agreements = (mask & (values[i].sign() == others.sign())).sum(dim=1).double()
                for key, result in (("sign_n", numbers), ("sign_agree", agreements)):
                    self.data[source + "_" + key][i, i + 1 :] += result
                    self.data[source + "_" + key][i + 1 :, i] += result
        # Distances, L1 and maximum differences are origin invariant: compute once.
        for i in range(self.count - 1):
            differences = final[i + 1 :] - final[i]
            for key, values in (
                ("distance", differences.square().sum(dim=1)),
                ("l1", differences.abs().sum(dim=1)),
                ("maximum", differences.abs().amax(dim=1)),
            ):
                if key == "maximum":
                    self.data[key][i, i + 1 :] = torch.maximum(self.data[key][i, i + 1 :], values)
                    self.data[key][i + 1 :, i] = self.data[key][i, i + 1 :]
                else:
                    self.data[key][i, i + 1 :] += values
                    self.data[key][i + 1 :, i] += values
        for group in self.groups:
            values = delta[group["indices"]]
            mean = values.mean(dim=0)
            block = torch.stack(
                (
                    values.square().sum(dim=1),
                    mean.square().sum().expand(values.shape[0]),
                    (values * mean).sum(dim=1),
                    (values - mean).square().sum(dim=1),
                ),
                dim=1,
            )
            self.shared[group["group_id"]] += block
        self.n += width

    def state(self):
        result = dict(
            n=self.n,
            groups=self.groups,
            data={k: v.cpu().tolist() for k, v in self.data.items()},
            shared={k: v.cpu().tolist() for k, v in self.shared.items()},
        )
        if not all(torch.isfinite(v).all() for v in self.data.values()):
            raise ValueError("Sufficient statistics overflowed")
        return result


def moments(state, source, a, b):
    data = state["data"]
    gram, centered = data[source + "_gram"], data[source + "_centered"]
    return VectorMoments(
        n=state["n"],
        qa=gram[a][a],
        qb=gram[b][b],
        dot=gram[a][b],
        qdiff=data["distance"][a][b],
        l1=data["l1"][a][b],
        maximum=data["maximum"][a][b],
        mean_a=data[source + "_mean"][a],
        mean_b=data[source + "_mean"][b],
        m2_a=centered[a][a],
        m2_b=centered[b][b],
        covariance=centered[a][b],
        sign_n=int(data[source + "_sign_n"][a][b]),
        sign_agree=int(data[source + "_sign_agree"][a][b]),
    )


PAIR_METRICS = (
    "l2_distance",
    "rmse",
    "cosine",
    "normalized_l2",
    "pearson",
    "sign_agreement",
    "l1_mean_distance",
    "max_abs_delta",
    "angle_over_pi",
)
DRIFT_METRICS = (
    "relative_l2",
    "delta_energy",
    "delta_rms",
    "cosine_final_base",
    "base_l2",
    "final_l2",
    "delta_mae",
    "max_abs_delta",
    "delta_mean",
)


def shared_values(values, size):
    energy, mean_energy, dot, distance = values
    if size < 2:
        raise ValueError("Leave-one-out requires at least two entities/languages")
    projection = projection_summary(energy, mean_energy, dot)
    loo_energy = max(0, (size * size * mean_energy + energy - 2 * size * dot) / (size - 1) ** 2)
    loo = projection_summary(energy, loo_energy, (size * dot - energy) / (size - 1))
    tolerance = 32 * torch.finfo(torch.float64).eps * max(energy, 1e-300)
    return dict(
        update_energy=energy,
        mean_update_energy=mean_energy,
        distance_to_mean=math.sqrt(max(0, distance)),
        projected_energy_fraction=projection["projection_fraction"],
        shared_cosine=projection["alignment_cosine"],
        specific_residual_energy=projection["direction_residual_energy"],
        leave_one_out_distance=math.sqrt(max(0, distance)) * size / (size - 1),
        leave_one_out_mean_energy=loo_energy,
        leave_one_out_projected_energy_fraction=loo["projection_fraction"],
        leave_one_out_cosine=loo["alignment_cosine"],
        dispersion_energy=distance,
        projection_roundoff_bound=tolerance,
    )


def moment_dict(value):
    return asdict(value)
