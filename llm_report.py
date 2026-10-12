import ollama
import json
import os

MEASUREMENT_PATH = "outputs/measurements.json"
OUTPUT_PATH = "outputs/intelligent_report.txt"

# Check measurements file
if not os.path.exists(MEASUREMENT_PATH):
    print("Error: measurements.json not found!")
    print("Run measurements.py first.")
    exit()

# Load measurements
with open(MEASUREMENT_PATH, "r", encoding="utf-8") as file:
    measurements = json.load(file)

print("Measurements loaded:")
print(measurements)

# Create prompt
prompt = f"""
You are an AI-assisted radiology report generation system.

Generate a structured radiology-style report for a Type A
Aortic Dissection segmentation project.

The segmentation was performed using a 3D U-Net.

Use ONLY the measurements provided below.
Do not invent any additional measurements or findings.

IMPORTANT:
- Dice and IoU are segmentation evaluation metrics.
- Do not call them diagnostic accuracy.
- Do not make a definitive medical diagnosis.
- Clearly state that the report requires review by a qualified radiologist.
- Recommendations are for clinical decision support only.

SEGMENTATION MEASUREMENTS:

Volume:
{measurements["volume_cm3"]} cm³

Volume:
{measurements["volume_mm3"]} mm³

Length:
{measurements["length_mm"]}

Width:
{measurements["width_mm"]}

Depth:
{measurements["depth_mm"]}

Maximum Diameter:
{measurements["maximum_diameter_mm"]}

Centroid:
{measurements["centroid_mm"]}

Generate the report with these sections:

1. AI RADIOLOGY REPORT
2. Study
3. AI Model
4. Segmentation Measurements
5. Findings
6. Recommendation
7. Disclaimer

Do not invent Dice or IoU values because they are not present
in the measurements file.
"""

print("\nSending actual measurements to Llama 3.2...")

# Call Llama
response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

intelligent_report = response["message"]["content"]

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Save report
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write(intelligent_report)

print("\n========================================")
print("INTELLIGENT REPORT GENERATED!")
print("========================================")
print(intelligent_report)
print("========================================")
print(f"Saved to: {OUTPUT_PATH}")