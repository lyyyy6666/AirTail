"""AirTail tail-aware PM2.5 forecasting."""

from airtail.models.pmg_rd.student import AirTailStudent
from airtail.models.pmg_rd.teacher import AirTailTeacher

__all__ = ["AirTailStudent", "AirTailTeacher"]
