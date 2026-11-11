import re
from datetime import datetime


def _parse_netcdf_hour_units(units):
    match = re.fullmatch(r"hours since (.+)", units.strip())
    if match is None:
        raise ValueError(f"unsupported NetCDF time units: {units!r}")
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00")).replace(tzinfo=None)

