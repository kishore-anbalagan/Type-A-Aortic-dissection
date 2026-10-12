import ollama


class RecommendationAgent:

    def __init__(self):

        self.model = "llama3.2"

    def generate_recommendation(
        self,
        measurements,
        evaluation
    ):

        print("\n===================================")
        print("      RECOMMENDATION AGENT")
        print("===================================")

        prompt = f"""
You are the Recommendation Agent in an Agentic AI
framework for AI-assisted Type A Aortic Dissection
CT angiography segmentation.

Your task is to analyze the ACTUAL segmentation
measurements and ACTUAL model evaluation metrics
provided below and generate an AI-assisted
clinical decision-support recommendation.

IMPORTANT RULES:

1. Use ONLY the information provided below.

2. Do NOT invent numerical values.

3. Do NOT modify numerical values.

4. Do NOT invent patient information.

5. Do NOT invent symptoms or medical history.

6. Do NOT make a definitive medical diagnosis.

7. Do NOT prescribe medication or treatment.

8. Do NOT recommend a specific medical procedure.

9. Do NOT infer false positives from the measurements.

10. Do NOT infer false negatives from the measurements.

11. Do NOT claim that the segmentation is incomplete
    unless this is explicitly supported by the provided
    evaluation information.

12. Do NOT claim that the segmentation contains false
    positives unless this is explicitly supported by
    the provided evaluation information.

13. Do NOT infer disease severity from volume, diameter,
    length, width, or depth alone.

14. Do NOT infer anatomical involvement that is not
    explicitly provided.

15. Do NOT describe the lesion as spherical, irregular,
    well-defined, poorly defined, etc., unless such
    information is explicitly provided.

16. Use Dice and IoU only to describe the agreement
    between the predicted segmentation and the reference
    segmentation.

17. If the Dice or IoU values indicate limited agreement,
    explain that the segmentation should be reviewed or
    verified.

18. The recommendation must remain an AI-assisted
    decision-support recommendation.

19. Final clinical interpretation must be performed by
    a qualified radiologist.

ACTUAL SEGMENTATION MEASUREMENTS:

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

TASK:

Analyze the segmentation quality using the Dice and IoU
scores.

Consider the segmentation measurements as quantitative
outputs only.

Generate a clear and concise recommendation based on
the actual results.

The recommendation should explain:

- The quality of agreement between predicted and
  reference segmentation.
- Whether additional verification of the segmentation
  is appropriate.
- The importance of qualified radiologist review for
  final clinical interpretation.

Do not provide a definitive diagnosis.

Do not provide treatment instructions.

Do not invent clinical findings.

Return ONLY the recommendation text.
"""

        print(
            "Sending actual measurements and evaluation "
            "results to Llama 3.2..."
        )

        try:

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            recommendation = response["message"]["content"]

        except Exception as e:

            print("\nError communicating with Llama 3.2:")
            print(e)

            return "Unable to generate the recommendation."

        print("\nLLM GENERATED RECOMMENDATION:\n")
        print(recommendation)

        return recommendation


if __name__ == "__main__":

    print("===================================")
    print("   LLM RECOMMENDATION AGENT")
    print("===================================")

    print("\nAgent initialized successfully.")
    print("Model:", "llama3.2")