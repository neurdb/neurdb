from datetime import datetime


def is_time_format(term):
    time_formats = [
        "%Y-%m-%d",  # Date format (e.g., 2021-01-01)
        "%H:%M",  # Time format (e.g., 12:30)
        "%Y-%m-%d %H:%M:%S",  # Date and time format (e.g., 2021-01-01 12:30:45)
    ]

    for fmt in time_formats:
        try:
            dt = datetime.strptime(term, fmt)
            return int(dt.timestamp())
        except ValueError:
            pass
    return None
