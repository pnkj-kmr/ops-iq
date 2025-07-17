from datetime import datetime


def encode_datetimes(obj):
    if isinstance(obj, dict):
        return {k: encode_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_datetimes(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj
