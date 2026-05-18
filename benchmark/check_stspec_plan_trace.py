#!/usr/bin/env python3
"""Summarize ST-Spec StepPlan diagnostics from raw engine trace JSON files.

This is a thin alias around check_multislo_result.py's plan-aware diagnostics;
it is intended for files produced by ``--engine-trace-out``.
"""

from __future__ import annotations

from check_multislo_result import main


if __name__ == "__main__":
    raise SystemExit(main())
