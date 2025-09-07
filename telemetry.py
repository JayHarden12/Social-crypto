import csv
import os
from datetime import datetime
from typing import Dict, Any


LOG_PATH = 'telemetry.log.csv'


def log_event(event: str, props: Dict[str, Any] | None = None) -> None:
    props = props or {}
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'event', *props.keys()])
        if not exists:
            writer.writeheader()
        row = {'timestamp': datetime.utcnow().isoformat(), 'event': event}
        row.update(props)
        writer.writerow(row)

