import json
import os


class MeasurementAgent:

    def __init__(self):

        self.measurement_file = "outputs/measurements.json"

    def get_measurements(self):

        print("\n===================================")
        print("        MEASUREMENT AGENT")
        print("===================================")

        if not os.path.exists(self.measurement_file):

            print("Error: measurements.json not found!")

            return None

        with open(
            self.measurement_file,
            "r"
        ) as file:

            measurements = json.load(file)

        print("Measurements successfully loaded.")

        print(
            "Volume      :",
            measurements.get("volume_cm3"),
            "cm³"
        )

        print(
            "Length      :",
            measurements.get("length_mm"),
            "mm"
        )

        print(
            "Width       :",
            measurements.get("width_mm"),
            "mm"
        )

        print(
            "Depth       :",
            measurements.get("depth_mm"),
            "mm"
        )

        print(
            "Maximum Diameter :",
            measurements.get("maximum_diameter_mm"),
            "mm"
        )

        return measurements


if __name__ == "__main__":

    agent = MeasurementAgent()

    agent.get_measurements()