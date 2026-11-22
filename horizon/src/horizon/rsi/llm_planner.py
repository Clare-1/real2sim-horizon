"""LLM planner selector with strict accounting and no silent degradation.

The advisor's minimal-validation table has an "LLM agent" row. This module provides it: one
decision per round, temperature 0, fixed model, full token accounting, and a hard failure when no
credential is present. It never falls back to a rule silently — a row that says "LLM" must have
actually called an LLM.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from .selector import ACTIONS, Decision, Observation, _guard

SYSTEM_PROMPT = (
    "You are choosing the next learning operation for a robot policy improvement loop. "
    "You see only deployment feedback: per-subgoal failure counts, the calibration residual, "
    "recovery triggers, the perturbation failure rate, the current success rate, the budget, "
    "and your own past decisions. You do NOT see the true cause of failure.\n"
    "Pick exactly one action from: " + ", ".join(ACTIONS) + ".\n"
    "Reply with JSON only: {\"action\": str, \"target\": str|null, \"reason\": str (<=25 words)}."
)


def build_user_prompt(obs: Observation) -> str:
    p, b = obs.profile, obs.budget
    payload = {
        "n_episodes": p.n_episodes,
        "sr": round(p.sr, 3),
        "per_subgoal_failure": p.per_subgoal_failure,
        "calibration_residual_mm": None if p.calibration_residual is None else round(p.calibration_residual * 1000, 2),
        "recovery_triggers": p.recovery_triggers,
        "perturbed_failure_rate": round(p.perturbed_failure_rate, 3),
        "budget": {"deploy_spent": b.deploy_spent, "deploy_total": b.deploy_total,
                   "labels_spent": b.labels_spent, "labels_total": b.labels_total,
                   "rounds_done": b.rounds_done, "rounds_total": b.rounds_total},
        "history": [{"action": h.action, "sr_before": round(h.sr_before, 3),
                     "sr_after": round(h.sr_after, 3)} for h in obs.history],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_decision(text: str) -> Decision:
    """Parse the model's reply; reject anything that is not one of the allowed actions."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"LLM reply has no JSON object: {text[:200]!r}")
    data = json.loads(match.group(0))
    action = str(data.get("action", "")).strip()
    if action not in ACTIONS:
        raise ValueError(f"LLM chose an action outside the action set: {action!r}")
    target = data.get("target")
    return Decision(action, None if target in (None, "", "null") else str(target),
                    str(data.get("reason", ""))[:200], 4)


@dataclass
class LLMPlannerSelector:
    """One LLM call per round. Raises if no credential: a missing key must stop the run."""

    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 512
    name: str = "llm"
    usage: list = field(default_factory=list)   # per-decision token records
    # Optional injected transport for pilots/tests: (system, user) -> (reply, usage).
    # When set, no network call is made and the API key is not required.
    client: Callable[[str, str], tuple[str, dict]] | None = None

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        if self.client is None and not self.available():
            raise RuntimeError(
                f"LLM selector needs {self.api_key_env}; refusing to substitute a rule planner. "
                "Set the key or remove 'llm' from the selector list.")
        reply, usage = self._call(build_user_prompt(obs))
        self.usage.append(usage)
        return parse_decision(reply)

    def _call(self, user_prompt: str) -> tuple[str, dict]:
        if self.client is not None:
            return self.client(SYSTEM_PROMPT, user_prompt)
        import urllib.request
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                     "x-api-key": os.environ[self.api_key_env]})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = {"model": self.model, "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                 "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                 "prompt_chars": len(user_prompt), "reply_chars": len(text)}
        return text, usage
