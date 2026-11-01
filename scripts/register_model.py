"""Moved to anomaly_inspector.pipelines.register, so the training image can run it from the package.

This name stays for the commands written before the move (evaluate_patchcore.py mentions it, and
editing that file would make DVC re-run every evaluation):
  uv run --env-file .env python scripts/register_model.py --category screw
"""

from anomaly_inspector.pipelines.register import main

if __name__ == "__main__":
    main()
