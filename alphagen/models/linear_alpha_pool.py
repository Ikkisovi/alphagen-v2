import math
from itertools import count
from typing import List, Optional, Tuple, Iterable, Dict, Any, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .alpha_pool import AlphaPoolBase
from ..data.calculator import AlphaCalculator, TensorAlphaCalculator
from ..data.expression import Expression, OutOfDataRangeError
from ..data.pool_update import PoolUpdate, AddRemoveAlphas
from ..utils.correlation import batch_pearsonr


class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, device)
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self._weights: np.ndarray = np.zeros(capacity + 1)
        self._mutual_ics: np.ndarray = np.identity(capacity + 1)
        self._extra_info = [None for _ in range(capacity + 1)]
        self._ic_lower_bound = -1. if ic_lower_bound is None else ic_lower_bound
        self.best_obj = -1.
        self.update_history: List[PoolUpdate] = []
        self._failure_cache: Set[str] = set()

    @property
    def weights(self) -> np.ndarray:
        "Get the weights of the linear model as a numpy array of shape (size,)."
        return self._weights[:self.size]

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        "Set the weights of the linear model with a numpy array of shape (size,)."
        assert value.shape == (self.size,), f"Invalid weights shape: {value.shape}"
        self._weights[:self.size] = value

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "exprs": self.exprs[:self.size],
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights),
            "best_ic_ret": self.best_ic_ret
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights)
        }

    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            # Log why this expression failed
            failure_reason = "unknown"
            if ic_ret is None:
                failure_reason = "ic_ret is None"
            elif ic_mut is None:
                failure_reason = "ic_mut is None (high correlation or below threshold)"
            elif np.isnan(ic_ret):
                failure_reason = "ic_ret is NaN"
            elif np.isnan(ic_mut).any():
                failure_reason = "ic_mut contains NaN"

            # Store failure stats for debugging
            if not hasattr(self, '_failure_stats'):
                self._failure_stats = {}
            self._failure_stats[failure_reason] = self._failure_stats.get(failure_reason, 0) + 1
            return 0.
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1
        old_pool: List[Expression] = self.exprs[:self.size]     # type: ignore
        self._add_factor(expr, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self.optimize()
            worst_idx = None
            if self.size > self.capacity:   # Need to remove one
                worst_idx = int(np.argmin(np.abs(new_weights)))
                # The one added this time is the worst, revert the changes
                if worst_idx == self.capacity:
                    self._pop(worst_idx)
                    self._failure_cache.add(str(expr))
                    return self.best_obj
            removed_idx = [worst_idx] if worst_idx is not None else []
            self.weights = new_weights
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=removed_idx,
                old_pool=old_pool,
                old_pool_ic=self.best_ic_ret,
                new_pool_ic=ic_ret
            ))
            if worst_idx is not None:
                self._pop(worst_idx)
        else:
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=[],
                old_pool=[],
                old_pool_ic=0.,
                new_pool_ic=ic_ret
            ))

        self._failure_cache = set()
        new_ic_ret, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic_ret, new_obj)
        return new_obj

    def force_load_exprs(self, exprs: List[Expression], weights: Optional[List[float]] = None) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        added = []
        for expr in exprs:
            if self.size >= self.capacity:
                break
            try:
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            except (OutOfDataRangeError, TypeError):
                continue
            if ic_ret is None or ic_mut is None:
                self._failure_cache.add(str(expr))
                continue
            if not math.isfinite(ic_ret):
                self._failure_cache.add(str(expr))
                continue
            if ic_mut and (not np.isfinite(np.array(ic_mut)).all()):
                self._failure_cache.add(str(expr))
                continue
            self._add_factor(expr, ic_ret, ic_mut)
            added.append(expr)
            assert self.size <= self.capacity
        if weights is not None:
            if len(weights) != self.size:
                raise ValueError(f"Invalid weights length: got {len(weights)}, expected {self.size}")
            self.weights = np.array(weights)
        else:
            self.weights = self.optimize()
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added,
            removed_idx=[],
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def calculate_ic_and_objective(self) -> Tuple[float, float]:
        ic = self.evaluate_ensemble()
        obj = self._calc_main_objective()
        if obj is None:
            obj = ic
        return ic, obj

    def _calc_main_objective(self) -> Optional[float]:
        "Get the main optimization objective, return None for the default (ensemble IC)."

    def _maybe_update_best(self, ic: float, obj: float) -> bool:
        # Track all objective values for debugging
        if not hasattr(self, '_obj_history'):
            self._obj_history = []
        self._obj_history.append((ic, obj))

        if obj <= self.best_obj:
            return False

        # Log when best is updated
        old_best_obj = self.best_obj
        old_best_ic = self.best_ic_ret
        self.best_obj = obj
        self.best_ic_ret = ic

        if not hasattr(self, '_best_update_count'):
            self._best_update_count = 0
        self._best_update_count += 1

        return True

    @abstractmethod
    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        "Optimize the weights of the linear model and return the new weights as a numpy array."

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        return calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights)      # type: ignore

    def evaluate_ensemble(self) -> float:
        if self.size == 0:
            return 0.
        return self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights)  # type: ignore

    @property
    def _under_thres_alpha(self) -> bool:
        if self._ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self._ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self._ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])     # type: ignore
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics
    
    def _get_extra_info(self, expr: Expression) -> Any:
        "Override this method to save extra data for a newly added expression."

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self._mutual_ics[i][n] = self._mutual_ics[n][i] = ic_mut[i]
        self._extra_info[n] = self._get_extra_info(expr)
        new_weight = max(ic_ret, 0.01) if n == 0 else self.weights.mean()
        self._weights[n] = new_weight   # Assign an initial weight
        self.size += 1

    def _pop(self, index_hint: Optional[int] = None) -> None:
        if self.size <= self.capacity:
            return
        index = int(np.argmin(np.abs(self.weights))) if index_hint is None else index_hint
        self._swap_idx(index, self.capacity)
        self.size = self.capacity

    def most_significant_indices(self, k: int) -> List[int]:
        if self.size == 0:
            return []
        ranks = (-np.abs(self.weights)).argsort().argsort()
        return [i for i in range(self.size) if ranks[i] < k]

    def leave_only(self, indices: Iterable[int]) -> None:
        "Leaves only the alphas at the given indices intact, and removes all others."
        self._failure_cache = set()
        indices = sorted(indices)
        for i, j in enumerate(indices):
            self._swap_idx(i, j)
        self.size = len(indices)

    def bulk_edit(self, removed_indices: Iterable[int], added_exprs: List[Expression]) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        removed_indices = set(removed_indices)
        remain = [i for i in range(self.size) if i not in removed_indices]
        old_exprs = {id(self.exprs[i]): i for i in range(self.size)}
        old_update_history_count = len(self.update_history)
        self.leave_only(remain)
        for e in added_exprs:
            self.try_new_expr(e)
        self.update_history = self.update_history[:old_update_history_count]
        new_exprs = {id(e): e for e in self.exprs[:self.size]}
        added_exprs = [e for e in self.exprs[:self.size] if id(e) not in old_exprs] # type: ignore
        removed_indices = list(sorted(i for eid, i in old_exprs.items() if eid not in new_exprs))
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added_exprs,
            removed_idx=removed_indices,
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def _swap_idx(self, i: int, j: int) -> None:
        if i == j:
            return

        def swap_in_list(lst, i: int, j: int) -> None:
            lst[i], lst[j] = lst[j], lst[i]

        swap_in_list(self.exprs, i, j)
        swap_in_list(self.single_ics, i, j)
        self._mutual_ics[:, [i, j]] = self._mutual_ics[:, [j, i]]
        self._mutual_ics[[i, j], :] = self._mutual_ics[[j, i], :]
        swap_in_list(self._weights, i, j)
        swap_in_list(self._extra_info, i, j)

    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debugging statistics to understand why best_ic_ret might not be improving."""
        stats = {
            'pool_size': self.size,
            'capacity': self.capacity,
            'eval_count': self.eval_cnt,
            'best_ic_ret': self.best_ic_ret,
            'best_obj': self.best_obj,
            'current_ensemble_ic': self.evaluate_ensemble() if self.size > 0 else None,
        }

        # Failure statistics
        if hasattr(self, '_failure_stats'):
            stats['failure_stats'] = self._failure_stats
            stats['total_failures'] = sum(self._failure_stats.values())
        else:
            stats['failure_stats'] = {}
            stats['total_failures'] = 0

        # Best update count
        if hasattr(self, '_best_update_count'):
            stats['best_updates'] = self._best_update_count
        else:
            stats['best_updates'] = 0

        # Recent objective history (last 10)
        if hasattr(self, '_obj_history') and len(self._obj_history) > 0:
            recent = self._obj_history[-10:]
            stats['recent_objectives'] = [
                {'ic': ic, 'obj': obj} for ic, obj in recent
            ]
            # Statistics on objectives
            all_objs = [obj for ic, obj in self._obj_history]
            stats['obj_stats'] = {
                'min': min(all_objs),
                'max': max(all_objs),
                'mean': sum(all_objs) / len(all_objs),
                'count': len(all_objs)
            }
        else:
            stats['recent_objectives'] = []
            stats['obj_stats'] = None

        # For CustomRewardAlphaPool, show objective components
        if hasattr(self, '_objective_components') and len(self._objective_components) > 0:
            recent_components = self._objective_components[-10:]
            stats['recent_objective_components'] = recent_components

        if hasattr(self, '_objective_errors') and len(self._objective_errors) > 0:
            stats['objective_errors'] = self._objective_errors[-10:]

        return stats

    def print_debug_stats(self) -> None:
        """Print debugging statistics in a human-readable format."""
        stats = self.get_debug_stats()
        print("\n" + "="*60)
        print("ALPHA POOL DEBUG STATISTICS")
        print("="*60)
        print(f"Pool Size: {stats['pool_size']}/{stats['capacity']}")
        print(f"Total Evaluations: {stats['eval_count']}")
        print(f"Best IC (ret): {stats['best_ic_ret']:.6f}")
        print(f"Best Objective: {stats['best_obj']:.6f}")

        # Fix: Move conditional outside format spec to avoid ValueError
        ensemble_ic_str = f"{stats['current_ensemble_ic']:.6f}" if stats['current_ensemble_ic'] is not None else 'N/A'
        print(f"Current Ensemble IC: {ensemble_ic_str}")
        print(f"Best Updates: {stats['best_updates']}")

        print(f"\nFailure Statistics (Total: {stats['total_failures']}):")
        for reason, count in sorted(stats['failure_stats'].items(), key=lambda x: -x[1]):
            pct = 100.0 * count / stats['eval_count'] if stats['eval_count'] > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

        if stats['obj_stats']:
            print(f"\nObjective Statistics ({stats['obj_stats']['count']} values):")
            print(f"  Min:  {stats['obj_stats']['min']:.6f}")
            print(f"  Max:  {stats['obj_stats']['max']:.6f}")
            print(f"  Mean: {stats['obj_stats']['mean']:.6f}")

        if stats['recent_objectives']:
            print(f"\nRecent Objectives (last {len(stats['recent_objectives'])}):")
            for i, entry in enumerate(stats['recent_objectives'], 1):
                print(f"  {i}. IC={entry['ic']:.6f}, Obj={entry['obj']:.6f}")

        if 'recent_objective_components' in stats:
            print(f"\nRecent Objective Components (last {len(stats['recent_objective_components'])}):")
            for i, comp in enumerate(stats['recent_objective_components'], 1):
                print(f"  {i}. IC={comp['ic']:.4f}, ICIR={comp['icir']:.4f}, "
                      f"Turnover={comp['turnover_penalty']:.4f}, Final={comp['final']:.4f}")

        if 'objective_errors' in stats:
            print(f"\nObjective Calculation Errors:")
            for err in stats['objective_errors']:
                print(f"  - {err}")

        print("="*60 + "\n")


class MseAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self._l1_alpha = l1_alpha

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha = self._l1_alpha
        if math.isclose(alpha, 0.):     # No L1 regularization, use the faster least-squares method
            return self._optimize_lstsq()
            
        ics_ret = torch.tensor(self.single_ics[:self.size], device=self.device)
        ics_mut = torch.tensor(self._mutual_ics[:self.size, :self.size], device=self.device)
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optim = torch.optim.Adam([weights], lr=lr)
    
        loss_ic_min = float("inf")
        best_weights = weights
        tolerance_count = 0
        for step in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()
    
            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1
    
            optim.zero_grad()
            loss.backward()
            optim.step()
    
            if loss_ic_min - loss_ic_curr > 1e-6:
                tolerance_count = 0
            else:
                tolerance_count += 1
    
            if loss_ic_curr < loss_ic_min:
                best_weights = weights
                loss_ic_min = loss_ic_curr
    
            if tolerance_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()
    
    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self._mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights


# Note: Currently the weights are only updated when the new IC is higher.
# It might be better to update the weights according to the actual objective,
# in this case the ICIR or the LCB of the IC.

class MeanStdAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        lcb_beta: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        """
        l1_alpha: the L1 regularization coefficient.
        lcb_beta: for optimizing the lower-confidence-bound: LCB = mean - beta * std, \
                  when this is None, optimize ICIR (mean / std) instead.
        """
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self.calculator: TensorAlphaCalculator
        self._l1_alpha = l1_alpha
        self._lcb_beta = lcb_beta

    def _get_extra_info(self, expr: Expression) -> Any:
        return self.calculator.evaluate_alpha(expr)
    
    def _calc_main_objective(self) -> float:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device)
        return self._calc_obj_impl(alpha_values, weights).item()
    
    def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target_value = self.calculator.target
        weighted = (weights[:, None, None] * alpha_values).sum(dim=0)
        ics = batch_pearsonr(weighted, target_value)
        mean, std = ics.mean(), ics.std()
        if self._lcb_beta is not None:
            return mean - self._lcb_beta * std
        else:
            return mean / std

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=lr)
    
        min_loss = float("inf")
        best_weights = weights
        tol_count = 0
        for step in count():
            obj = self._calc_obj_impl(alpha_values, weights)
            loss_l1 = torch.norm(weights, p=1)
            loss = self._l1_alpha * loss_l1 - obj   # Maximize the objective
            curr_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if min_loss - curr_loss > 1e-6:
                tol_count = 0
            else:
                tol_count += 1
    
            if curr_loss < min_loss:
                best_weights = weights
                min_loss = curr_loss
    
            if tol_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()

class CustomRewardAlphaPool(MeanStdAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        lcb_beta: Optional[float] = None,
        turnover_penalty_coeff: float = 0.0,  # FIX 2: Default 0 (disabled) like original
        turnover_rebalance_horizon: Optional[int] = None,
        turnover_top_k: Optional[int] = None,
        turnover_top_k_ratio: float = 0.1,
        use_icir: bool = False,  # FIX 2: Add switch for ICIR (default False = use IC only)
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, ic_lower_bound, l1_alpha, lcb_beta, device)
        self.turnover_penalty_coeff = turnover_penalty_coeff
        self.turnover_rebalance_horizon = max(1, int(turnover_rebalance_horizon)) if turnover_rebalance_horizon else 1
        self.turnover_top_k = turnover_top_k
        self.turnover_top_k_ratio = turnover_top_k_ratio
        self.use_icir = use_icir  # FIX 2: Switch for ICIR

    def _compute_selection_turnover(self, weighted_alpha: torch.Tensor) -> float:
        horizon = max(1, int(self.turnover_rebalance_horizon))
        periods, num_assets = weighted_alpha.shape

        if periods <= horizon or num_assets == 0:
            return 0.0

        if self.turnover_top_k is not None and self.turnover_top_k > 0:
            top_k = min(self.turnover_top_k, num_assets)
        else:
            ratio = self.turnover_top_k_ratio if self.turnover_top_k_ratio > 0 else 0.1
            top_k = max(1, min(num_assets, int(round(num_assets * ratio))))

        if top_k == 0:
            return 0.0

        turnover_samples: List[float] = []
        fill_value = torch.full(
            (num_assets,),
            torch.finfo(weighted_alpha.dtype).min,
            device=self.device,
            dtype=weighted_alpha.dtype,
        )

        for start in range(0, periods - horizon, horizon):
            current_scores = weighted_alpha[start]
            future_scores = weighted_alpha[start + horizon]

            current_valid = ~torch.isnan(current_scores)
            future_valid = ~torch.isnan(future_scores)

            # Skip if we do not have any valid scores in either snapshot
            if not current_valid.any() or not future_valid.any():
                continue

            effective_k = min(top_k, int(current_valid.sum().item()), int(future_valid.sum().item()))
            if effective_k == 0:
                continue

            current_filled = torch.where(current_valid, current_scores, fill_value)
            future_filled = torch.where(future_valid, future_scores, fill_value)

            current_indices = torch.topk(current_filled, k=effective_k, largest=True).indices.detach().cpu().tolist()
            future_indices = torch.topk(future_filled, k=effective_k, largest=True).indices.detach().cpu().tolist()

            current_set = set(current_indices)
            future_set = set(future_indices)

            intersection = len(current_set & future_set)
            turnover_samples.append(1.0 - intersection / float(effective_k))

        if not turnover_samples:
            return 0.0

        return float(sum(turnover_samples) / len(turnover_samples))

    def _calc_main_objective(self) -> Optional[float]:
        """
        FIX 2: Configurable reward function with switches.
        - If use_icir=False and turnover_penalty_coeff=0: Returns None (uses default IC, like original MseAlphaPool)
        - If use_icir=True: Adds ICIR component
        - If turnover_penalty_coeff>0: Subtracts turnover penalty
        """
        if self.size == 0:
            return None

        # FIX 2: If both switches are disabled, use default IC (like original alphagen)
        if not self.use_icir and self.turnover_penalty_coeff <= 0:
            return None  # Fall back to default ensemble IC

        try:
            alpha_values = torch.stack(self._extra_info[:self.size])
            weights = torch.tensor(self.weights, device=self.device)

            # Calculate Ensemble IC
            target_value = self.calculator.target
            weighted_alpha = (weights[:, None, None] * alpha_values).sum(dim=0)
            ics = batch_pearsonr(weighted_alpha, target_value)

            # Check for NaN in IC calculation - this is critical
            if torch.isnan(ics).any():
                return None

            ensemble_ic = ics.mean().item()
            if not math.isfinite(ensemble_ic):
                return None

            # Start with IC as base objective
            final_objective = ensemble_ic

            # FIX 2: Optionally add ICIR component
            ensemble_icir = 0.0
            if self.use_icir:
                ic_mean = ics.mean()
                ic_std = ics.std()

                # If std is too small or NaN, we can't calculate ICIR reliably
                if torch.isnan(ic_std) or ic_std.item() < 1e-6:
                    ensemble_icir = 0.0
                else:
                    ensemble_icir = (ic_mean / ic_std).item()
                    if not math.isfinite(ensemble_icir):
                        ensemble_icir = 0.0

                final_objective += ensemble_icir

            # FIX 2: Optionally add turnover penalty
            turnover = 0.0
            turnover_penalty = 0.0
            if self.turnover_penalty_coeff > 0 and weighted_alpha.shape[0] > 1:
                turnover = self._compute_selection_turnover(weighted_alpha)
                if math.isfinite(turnover):
                    turnover_penalty = self.turnover_penalty_coeff * turnover
                    final_objective -= turnover_penalty
                else:
                    turnover = 0.0

            # Store components for debugging
            if not hasattr(self, '_objective_components'):
                self._objective_components = []
            self._objective_components.append({
                'ic': ensemble_ic,
                'icir': ensemble_icir,
                'turnover': turnover,
                'turnover_penalty': turnover_penalty,
                'final': final_objective
            })

            return final_objective
        except Exception as e:
            # If anything goes wrong, return None to fall back to IC
            if not hasattr(self, '_objective_errors'):
                self._objective_errors = []
            self._objective_errors.append(str(e))
            return None
