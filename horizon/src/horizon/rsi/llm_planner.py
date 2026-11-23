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
import time
from dataclasses import dataclass, field
from typing import Callable

from .selector import ACTIONS, Decision, Observation, _guard

SYSTEM_PROMPT_TEMPLATE = (
    "You are choosing the next learning operation for a robot policy improvement loop. "
    "You see only deployment feedback: per-subgoal failure counts, the calibration residual, "
    "recovery triggers, the perturbation failure rate, the current success rate, the budget, "
    "and your own past decisions. You do NOT see the true cause of failure.\n"
    "Pick exactly one action from: {actions}.\n"
    "Reply with JSON only: {{\"action\": str, \"target\": str|null, \"reason\": str (<=25 words)}}."
)
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(actions=", ".join(ACTIONS))


def system_prompt(allowed_actions=None) -> str:
    """Prompt with a restricted action list (agent ablation: 'no specialist tools')."""
    if allowed_actions is None:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT_TEMPLATE.format(actions=", ".join(allowed_actions))


def build_user_prompt(obs: Observation, memory: tuple = ()) -> str:
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
    if memory:
        # Past-scenario lessons supplied by the operator; they must not contain the true cause.
        payload["memory"] = list(memory)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_decision(text: str, allowed=None) -> Decision:
    """Parse the model's reply; reject anything that is not one of the allowed actions."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"LLM reply has no JSON object: {text[:200]!r}")
    data = json.loads(match.group(0))
    action = str(data.get("action", "")).strip()
    if action not in ACTIONS:
        raise ValueError(f"LLM chose an action outside the action set: {action!r}")
    if allowed is not None and action not in allowed:
        raise ValueError(f"LLM chose an action withheld from it ({action!r}; allowed={tuple(allowed)})")
    target = data.get("target")
    return Decision(action, None if target in (None, "", "null") else str(target),
                    str(data.get("reason", ""))[:200], 4)


@dataclass
class LLMPlannerSelector:
    """One LLM call per round. Raises if no credential: a missing key must stop the run."""

    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    # "anthropic" posts to /v1/messages; "openai" posts to an OpenAI-compatible
    # /chat/completions endpoint (OpenAI, DeepSeek, vLLM, most proxies). A key that starts with
    # "sk-" and has no "ant-" marker belongs to the latter.
    provider: str = "anthropic"
    base_url: str = ""
    max_tokens: int = 512
    name: str = "llm"
    usage: list = field(default_factory=list)   # per-decision token/cost records
    # Optional injected transport for pilots/tests: (system, user) -> (reply, usage).
    # When set, no network call is made and the API key is not required.
    client: Callable[[str, str], tuple[str, dict]] | None = None
    # Cost/robustness guards. Prices are per million tokens; set them to the current model price.
    max_retries: int = 3
    retry_backoff: float = 2.0
    max_calls: int | None = None
    max_usd: float | None = None
    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0
    # Past-scenario lessons for the "LLM + experience" row; never include the true cause.
    memory: tuple = ()
    # Agent ablation: when set, only these actions are offered and anything else is rejected.
    allowed_actions: tuple | None = None

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    @property
    def spent_usd(self) -> float:
        return sum(float(u.get("usd", 0.0)) for u in self.usage)

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        if self.client is None and not self.available():
            raise RuntimeError(
                f"LLM selector needs {self.api_key_env}; refusing to substitute a rule planner. "
                "Set the key or remove 'llm' from the selector list.")
        if self.max_calls is not None and len(self.usage) >= self.max_calls:
            raise RuntimeError(f"LLM call cap reached ({self.max_calls}); stopping instead of overspending")
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise RuntimeError(f"LLM cost cap reached (${self.max_usd:.2f}); stopping instead of overspending")
        reply, usage = self._call(build_user_prompt(obs, self.memory))
        self.usage.append(usage)
        return parse_decision(reply, self.allowed_actions)

    def _call(self, user_prompt: str) -> tuple[str, dict]:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._once(user_prompt)
            except (OSError, TimeoutError) as exc:  # URLError and socket errors are OSError
                last = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last}")

    def _once(self, user_prompt: str) -> tuple[str, dict]:
        if self.client is not None:
            return self.client(system_prompt(self.allowed_actions), user_prompt)
        if self.provider == "openai":
            return self._once_openai(user_prompt)
        import urllib.request
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
            "system": system_prompt(self.allowed_actions),
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                     "x-api-key": os.environ[self.api_key_env]})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = "".join(block.get("text", "") for block in data.get("content", []))
        inp = int(data.get("usage", {}).get("input_tokens", 0))
        out = int(data.get("usage", {}).get("output_tokens", 0))
        usage = {"model": self.model, "input_tokens": inp, "output_tokens": out,
                 "usd": (inp * self.price_in_per_mtok + out * self.price_out_per_mtok) / 1e6,
                 "prompt_chars": len(user_prompt), "reply_chars": len(text)}
        return text, usage

    def _once_openai(self, user_prompt: str) -> tuple[str, dict]:
        """OpenAI-compatible chat completion. Same accounting, same hard failure on a bad reply."""
        import urllib.request
        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
            "messages": [{"role": "system", "content": system_prompt(self.allowed_actions)},
                         {"role": "user", "content": user_prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions", data=body,
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {os.environ[self.api_key_env]}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM reply had no choices: {json.dumps(data)[:300]}")
        text = str((choices[0].get("message") or {}).get("content") or "")
        usage_raw = data.get("usage") or {}
        inp = int(usage_raw.get("prompt_tokens", 0))
        out = int(usage_raw.get("completion_tokens", 0))
        usage = {"model": data.get("model", self.model), "input_tokens": inp, "output_tokens": out,
                 "usd": (inp * self.price_in_per_mtok + out * self.price_out_per_mtok) / 1e6,
                 # Some OpenAI-compatible proxies bill per call and report it directly; keep it
                 # alongside our price-table estimate so the two can be reported separately.
                 "provider_cost": usage_raw.get("cost"),
                 "prompt_chars": len(user_prompt), "reply_chars": len(text),
                 "provider": "openai", "base_url": base}
        return text, usage
