"""
exporters.py

Export utilities for mocap data.
"""

import json
import csv


class MocapExporter:

    def export_json(self, frames, filepath):

        with open(filepath, "w") as f:
            json.dump(frames, f, indent=2)

    def export_csv(self, frames, filepath):

        with open(filepath, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["timestamp", "pose"])

            for frame in frames:

                writer.writerow([
                    frame["timestamp"],
                    frame["pose"]
                ])
