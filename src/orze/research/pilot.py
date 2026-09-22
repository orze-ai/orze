"""Optional, experiment-authored pilot before expensive method training.

Call only inside the caller's authorized research worker. This module grants no
execution permission, runs no model request, adds no data, and assigns no score.
The pilot chooses its own test and decision rule; it is not a fixed search menu.
Disabled by default until prospective time-to-accepted-goal validation.
"""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class TrainingResult:
    kind: str
    model: object
    findings: dict


def _findings(value, max_bytes):
    if not isinstance(value, dict):
        raise ValueError('findings must be an object')
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode('utf-8')) > max_bytes:
        raise ValueError('findings exceed the declared byte limit')
    # Snapshot observations before later training can mutate a shared dictionary.
    return json.loads(encoded)


def run_training(candidate, api, *, allow_pilot=False, max_findings_bytes=10000):
    """Train once, or return an unscored diagnostic after a bounded pilot.

    Existing build/train/predict methods retain their contract. An opt-in method
    may additionally define pilot(api) -> {'continue': bool, 'findings': dict,
    'model': optional partially trained object}. The API and outer container's
    resource/data limits are unchanged. A stopped pilot never reaches build,
    train, persistence or prediction; the caller must record it as an unscored
    analysis, even if its findings contain a self-reported improvement.

    Continuing may reuse the exact partial model, avoiding duplicate training.
    Absent a partial model, build(api) runs normally. Fresh held-out prediction
    loads the saved final model and MUST NOT invoke this training helper again.
    A pilot failure propagates as an execution failure, not negative science.
    """
    if type(allow_pilot) is not bool:
        raise ValueError('allow_pilot must be an explicit boolean')
    if type(max_findings_bytes) is not int or max_findings_bytes < 1:
        raise ValueError('findings byte limit must be a positive integer')
    pilot = getattr(candidate, 'pilot', None) if allow_pilot else None
    if pilot is None:
        model = candidate.build(api)
        findings = _findings(candidate.train(api, model), max_findings_bytes)
        return TrainingResult('method', model, findings)
    if not callable(pilot):
        raise ValueError('pilot must be callable')
    decision = pilot(api)
    if (not isinstance(decision, dict) or
            not {'continue', 'findings'} <= set(decision) <= {'continue', 'findings', 'model'} or
            type(decision['continue']) is not bool):
        raise ValueError('pilot requires a boolean continue, findings, and optional model')
    observations = _findings(decision['findings'], max_findings_bytes)
    if not decision['continue']:
        findings = _findings({'pilot': observations, 'continued': False}, max_findings_bytes)
        return TrainingResult('analysis', None, findings)
    model = decision['model'] if 'model' in decision else candidate.build(api)
    training = _findings(candidate.train(api, model), max_findings_bytes)
    findings = _findings({'pilot': observations, 'continued': True,
                         'training': training}, max_findings_bytes)
    return TrainingResult('method', model, findings)
