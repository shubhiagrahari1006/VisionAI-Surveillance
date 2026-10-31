"""
VisionAI - Object & Bidirectional Line-Crossing Counting
Supports class frequency counting, bidirectional boundary monitoring,
and live campus traffic occupancy summaries.
"""

from collections import Counter


def count_objects(detections):
    """Count detections by object class."""
    return dict(
        Counter(
            item["class"]
            for item in detections
            if item.get("class")
        )
    )


class LineCrossingCounter:
    """
    Bidirectional Line-Crossing Counter.
    Monitors objects crossing a horizontal boundary line and determines trajectory:
    - 'inbound'  : vehicle moving top-to-bottom across the line (Campus Entry).
    - 'outbound' : vehicle moving bottom-to-top across the line (Campus Exit).
    """

    def __init__(self, line_y, allowed_classes=None):
        self.line_y = float(line_y)
        self.allowed_classes = (
            set(allowed_classes) if allowed_classes else None
        )
        self.previous_positions = {}
        self.counted_ids = set()
        self.counts = Counter()               # Total crossings by class
        self.inbound_counts = Counter()       # Inbound (Entry) crossings
        self.outbound_counts = Counter()      # Outbound (Exit) crossings
        self.trajectories = {}                # track_id -> 'inbound' | 'outbound'

    def update(self, track_id, object_class, center_y):
        """
        Update tracked object position and detect line crossing with direction.

        Returns:
            direction ('inbound' | 'outbound') if a new crossing occurred, else None.
            (Note: Non-empty strings evaluate to True in boolean contexts, e.g. `if crossed:`).
        """
        if track_id is None:
            return None

        if (
            self.allowed_classes is not None
            and object_class not in self.allowed_classes
        ):
            return None

        track_id = int(track_id)
        center_y = float(center_y)

        previous_y = self.previous_positions.get(track_id)
        self.previous_positions[track_id] = center_y

        if previous_y is None or track_id in self.counted_ids:
            return None

        direction = None
        # Moving top to bottom (Increasing Y in image coordinates -> Entry)
        if previous_y < self.line_y <= center_y:
            direction = "inbound"
        # Moving bottom to top (Decreasing Y in image coordinates -> Exit)
        elif previous_y > self.line_y >= center_y:
            direction = "outbound"

        if direction:
            self.counted_ids.add(track_id)
            self.counts[object_class] += 1
            if direction == "inbound":
                self.inbound_counts[object_class] += 1
            else:
                self.outbound_counts[object_class] += 1
            self.trajectories[track_id] = direction
            return direction

        return None

    def get_counts(self):
        """Return total crossing counts by vehicle class."""
        return dict(self.counts)

    def get_inbound_counts(self):
        """Return inbound (entry) crossing counts by vehicle class."""
        return dict(self.inbound_counts)

    def get_outbound_counts(self):
        """Return outbound (exit) crossing counts by vehicle class."""
        return dict(self.outbound_counts)

    def get_trajectory(self, track_id):
        """Return the crossing trajectory of a specific track ID if known."""
        return self.trajectories.get(int(track_id)) if track_id is not None else None

    def get_summary(self):
        """Return full bidirectional traffic summary dictionary."""
        inbound_total = sum(self.inbound_counts.values())
        outbound_total = sum(self.outbound_counts.values())
        total = sum(self.counts.values())
        return {
            "inbound_total": inbound_total,
            "outbound_total": outbound_total,
            "total_crossings": total,
            "net_inside": max(0, inbound_total - outbound_total),
            "inbound_by_class": dict(self.inbound_counts),
            "outbound_by_class": dict(self.outbound_counts),
        }
