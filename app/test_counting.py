
from counting import count_objects

sample_detections = [
    {"class": "car"},
    {"class": "person"},
    {"class": "car"},
    {"class": "truck"},
    {"class": "person"},
    {"class": "car"},
]

counts = count_objects(sample_detections)

print("Object Counting Test")
print("--------------------")

for object_name, total in counts.items():
    print(f"{object_name}: {total}")

print(f"Total objects: {sum(counts.values())}")
