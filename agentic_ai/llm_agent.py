import ollama


class LLMAgent:

    def __init__(self):

        self.model = "llama3.2"

    def generate_report(
        self,
        measurements,
        evaluation
    ):

        print("\n===================================")
        print("           LLM AGENT")
        print("===================================")

        prompt = f"""
You are the LLM Report Agent in an Agentic AI framework
for AI-assisted analysis of Type A Aortic Dissection
CT angiography segmentation.

Generate a factual AI-assisted report using ONLY the
actual measurements and evaluation values provided below.

IMPORTANT RULES:

1. Do NOT invent findings.
2. Do NOT claim that a lesion is spherical, well-defined,
   irregular, or any other shape unless that information
   is explicitly provided.
3. Do NOT claim that Type A Aortic Dissection is confirmed
   from segmentation measurements alone.
4. Do NOT invent patient information.
5. Do NOT invent clinical symptoms.
6. Do NOT invent treatment.
7. Do NOT change numerical values.
8. Do NOT describe the Dice or IoU as "good", "moderate",
   or "high" without appropriate justification.
9. Explain that the current segmentation metrics indicate
   limited segmentation agreement when appropriate.
10. The report is AI-assisted clinical decision support,
    not a definitive diagnosis.
11. Final clinical interpretation must be performed by
    a qualified radiologist.

ACTUAL MEASUREMENTS:

Segmented Voxels:
{measurements.get("segmented_voxels")}

Voxel Spacing:
{measurements.get("voxel_spacing_mm")}

Volume:
{measurements.get("volume_cm3")} cm3

Volume:
{measurements.get("volume_mm3")} mm3

Length:
{measurements.get("length_mm")} mm

Width:
{measurements.get("width_mm")} mm

Depth:
{measurements.get("depth_mm")} mm

Maximum Diameter:
{measurements.get("maximum_diameter_mm")} mm

Centroid:
{measurements.get("centroid_mm")}

MODEL EVALUATION:

Dice Score:
{evaluation.get("dice_score")}

IoU Score:
{evaluation.get("iou_score")}

Generate the report using these sections:

1. AI Segmentation Summary
2. Measurements
3. Evaluation
4. Findings
5. Recommendation
6. Disclaimer

For Findings, only describe what can be directly supported
by the supplied segmentation measurements.

For Evaluation, explain that the Dice and IoU values represent
agreement between the predicted segmentation and reference
segmentation. Avoid unsupported claims about clinical accuracy.

For Recommendation, explain what additional review or
verification is appropriate based on the segmentation results.

Do not provide treatment instructions.

Return only the report.
"""

        print(
            "Sending actual results to Llama 3.2..."
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        report = response["message"]["content"]

        print("\nLLM REPORT GENERATED\n")

        print(report)

        return report