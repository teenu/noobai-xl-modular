"""DoRA schedule parsing and generation."""

from typing import Optional, Tuple, List
from config import logger


def parse_manual_dora_schedule(schedule_input: Optional[str], num_steps: int) -> Tuple[Optional[List[int]], Optional[str]]:
    """Parse and validate manual DoRA schedule from CSV string."""
    if not schedule_input or not isinstance(schedule_input, str) or not schedule_input.strip():
        return None, None

    if len(schedule_input) > 10000:
        return None, "Manual DoRA schedule too long (max 10000 characters) - DoRA will be OFF for all steps"

    if num_steps <= 0:
        return None, "Invalid number of steps"

    try:
        parts = [p.strip() for p in schedule_input.split(',')]

        schedule = []
        had_invalid_tokens = False

        for part in parts:
            if part == '1':
                schedule.append(1)
            elif part == '0':
                schedule.append(0)
            else:
                schedule.append(0)
                had_invalid_tokens = True

        if not schedule:
            return None, "Manual DoRA schedule is empty or malformed - DoRA will be OFF for all steps"

        warning_parts = []
        if had_invalid_tokens:
            warning_parts.append("some entries were invalid (non-0/1) and treated as 0")

        if len(schedule) < num_steps:
            diff = num_steps - len(schedule)
            schedule.extend([0] * diff)
            warning_parts.append(f"{diff} missing step(s) set to OFF")
        elif len(schedule) > num_steps:
            diff = len(schedule) - num_steps
            schedule = schedule[:num_steps]
            warning_parts.append(f"{diff} extra step(s) ignored")

        warning = f"Manual DoRA schedule: {', '.join(warning_parts)}" if warning_parts else None
        return schedule, warning

    except Exception as e:
        logger.warning(f"Failed to parse manual DoRA schedule: {e}")
        return None, f"Manual DoRA schedule is malformed ({str(e)}) - DoRA will be OFF for all steps"


def generate_standard_schedule(num_steps: int) -> List[int]:
    """Generate standard toggle schedule: ON,OFF,ON,OFF throughout all steps."""
    if num_steps <= 0:
        return []
    return [1 if i % 2 == 0 else 0 for i in range(num_steps)]


def generate_smart_schedule(num_steps: int) -> List[int]:
    """Generate smart toggle schedule: ON,OFF through step 20, then ON for remainder."""
    if num_steps <= 0:
        return []
    schedule = []
    for i in range(num_steps):
        if i <= 19:
            schedule.append(1 if i % 2 == 0 else 0)
        else:
            schedule.append(1)
    return schedule
