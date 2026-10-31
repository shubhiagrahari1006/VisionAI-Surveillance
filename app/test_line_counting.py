"""
Unit test for Bidirectional Line-Crossing Counter.
"""

from counting import LineCrossingCounter

counter = LineCrossingCounter(
    line_y=100,
    allowed_classes=["car", "truck", "bus"],
)

print("Bidirectional Line-Crossing Counting Test")
print("-----------------------------------------")

# Car ID 1 approaches from above and moves down (Inbound)
print("Car before line:", counter.update(1, "car", 80))
print("Car crosses line (Inbound):", counter.update(1, "car", 120))
print("Car after crossing (No duplicate):", counter.update(1, "car", 140))

# Truck ID 2 approaches from below and moves up (Outbound)
print("Truck before line:", counter.update(2, "truck", 130))
print("Truck crosses line (Outbound):", counter.update(2, "truck", 70))
print("Truck after crossing (No duplicate):", counter.update(2, "truck", 50))

# Person ID 3 crosses, but pedestrians are excluded
print("Person crossing (Filtered):", counter.update(3, "person", 120))

print("\nSummary:", counter.get_summary())
