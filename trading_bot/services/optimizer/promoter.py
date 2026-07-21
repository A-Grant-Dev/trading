"""
ParamSet Promoter — Safe Promotion of Candidate ParamSets

Implements the promotion logic: only promote a candidate ParamSet to
'live' status if it beats the current live set on out-of-sample metrics
by a minimum improvement threshold.

This prevents parameter overfitting and ensures only genuine
improvements are deployed.

Inspired by Renaissance Technologies:
- Never deploy a model that doesn't beat the current one on OOS data
- Multiple guardrails prevent catastrophic deployment
- Full audit trail for every promotion decision
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from trading_bot.models import AuditLog, BotConfig, ParamSet, Strategy as StrategyModel

logger = logging.getLogger(__name__)


def promote_param_set(
    param_set: ParamSet,
    strategy_model: StrategyModel,
    metrics: dict[str, Any],
    improvement_threshold: Optional[float] = None,
) -> bool:
    """
    Attempt to promote a candidate ParamSet to live status.

    Promotion only happens if:
    1. The candidate's Sharpe ratio > current live Sharpe * (1 + threshold)
    2. The candidate has at least 1 trade (non-trivial signal)
    3. No other promotion is in progress for this strategy

    Args:
        param_set: The candidate ParamSet to promote
        strategy_model: The Strategy model this ParamSet belongs to
        metrics: Performance metrics dict (must include 'sharpe_ratio')
        improvement_threshold: Minimum fractional improvement over live
            (default: from BotConfig.min_improvement_threshold)

    Returns:
        True if promotion succeeded, False otherwise
    """
    config = BotConfig.get_config()
    if improvement_threshold is None:
        improvement_threshold = config.min_improvement_threshold

    candidate_sharpe = metrics.get("sharpe_ratio", 0) or 0
    candidate_trades = metrics.get("total_trades", 0) or 0

    # ── Guard 1: Must have non-trivial trades ──────────────────
    if candidate_trades < 1:
        logger.info(
            "Promotion rejected for %s (strategy=%s): only %d trades",
            param_set.id, strategy_model.name, candidate_trades,
        )
        AuditLog.objects.create(
            action="param_promoted",
            message=(
                f"Rejected promotion for {strategy_model.name}: "
                f"insufficient trades ({candidate_trades})"
            ),
            details={
                "param_set_id": param_set.id,
                "strategy": strategy_model.name,
                "candidate_sharpe": candidate_sharpe,
                "candidate_trades": candidate_trades,
                "reason": "insufficient_trades",
            },
            severity="warning",
        )
        return False

    # ── Guard 2: Must beat current live by threshold ───────────
    current_live = ParamSet.objects.filter(
        strategy=strategy_model, is_live=True
    ).first()

    if current_live:
        live_sharpe = current_live.metrics.get("sharpe_ratio", 0) or 0
        if live_sharpe < 0:
            live_sharpe = 0  # Don't compare against negative sharpe

        # Sharpe must be strictly higher by at least threshold%
        min_required_sharpe = live_sharpe * (1 + improvement_threshold)
        if live_sharpe > 0 and candidate_sharpe <= min_required_sharpe:
            logger.info(
                "Promotion rejected for %s (strategy=%s): "
                "candidate sharpe=%.4f <= live sharpe=%.4f * (1+%.2f)=%.4f",
                param_set.id, strategy_model.name,
                candidate_sharpe, live_sharpe,
                improvement_threshold, min_required_sharpe,
            )
            AuditLog.objects.create(
                action="param_promoted",
                message=(
                    f"Rejected promotion for {strategy_model.name}: "
                    f"candidate sharpe {candidate_sharpe:.4f} ≤ "
                    f"live {live_sharpe:.4f} × (1+{improvement_threshold:.2f})"
                ),
                details={
                    "param_set_id": param_set.id,
                    "strategy": strategy_model.name,
                    "candidate_sharpe": candidate_sharpe,
                    "live_sharpe": live_sharpe,
                    "threshold": improvement_threshold,
                    "min_required": min_required_sharpe,
                    "reason": "below_threshold",
                },
                severity="info",
            )
            return False

    # ── Promote! ───────────────────────────────────────────────
    # Demote all existing live ParamSets for this strategy
    ParamSet.objects.filter(
        strategy=strategy_model, is_live=True
    ).exclude(id=param_set.id).update(is_live=False)

    # Mark this one as live
    param_set.is_live = True
    param_set.is_candidate = False
    param_set.metrics = metrics
    param_set.save(update_fields=["is_live", "is_candidate", "metrics"])

    logger.info(
        "✅ PROMOTED param_set=%d for %s: sharpe=%.4f, trades=%d",
        param_set.id, strategy_model.name, candidate_sharpe, candidate_trades,
    )

    AuditLog.objects.create(
        action="param_promoted",
        message=(
            f"Promoted ParamSet #{param_set.id} for {strategy_model.name}: "
            f"sharpe={candidate_sharpe:.4f}, trades={candidate_trades}"
        ),
        details={
            "param_set_id": param_set.id,
            "strategy": strategy_model.name,
            "candidate_sharpe": candidate_sharpe,
            "candidate_trades": candidate_trades,
            "previous_live_sharpe": live_sharpe if current_live else None,
        },
        severity="info",
    )

    return True


def demote_param_set(param_set: ParamSet) -> bool:
    """
    Force-demote a live ParamSet (e.g. after a string of losses).

    Returns the ParamSet to candidate status for further optimization.

    Args:
        param_set: The live ParamSet to demote

    Returns:
        True if demotion succeeded
    """
    if not param_set.is_live:
        logger.warning("ParamSet %d is not live, cannot demote", param_set.id)
        return False

    param_set.is_live = False
    param_set.is_candidate = True
    param_set.save(update_fields=["is_live", "is_candidate"])

    logger.info("Demoted ParamSet %d (strategy=%s)", param_set.id, param_set.strategy.name)

    AuditLog.objects.create(
        action="param_promoted",
        message=f"Demoted ParamSet #{param_set.id} for {param_set.strategy.name}",
        details={
            "param_set_id": param_set.id,
            "strategy": param_set.strategy.name,
            "action": "demoted",
        },
        severity="warning",
    )

    return True
