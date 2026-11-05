"""Compatibility API for the standard-library-only login-host scheduler adapter.

Use scripts/picasso/submit.sh on Picasso: the scientific CLI may run inside a
container whose PATH does not contain the host's scheduler commands.
"""

from weight_analysis_scheduler.submission import parse_job_id, submit

__all__ = ["parse_job_id", "submit"]
