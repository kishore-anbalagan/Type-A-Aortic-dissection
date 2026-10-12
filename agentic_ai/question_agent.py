import json
import os
import ollama


class QuestionAgent:

    def __init__(self):

        self.measurement_file = "outputs/measurements.json"
        self.evaluation_file = "outputs/evaluation.json"

        self.model = "llama3.2"

    def load_data(self):

        # Check measurement file
        if not os.path.exists(self.measurement_file):

            print("Error: measurements.json not found!")

            return None, None

        # Check evaluation file
        if not os.path.exists(self.evaluation_file):

            print("Error: evaluation.json not found!")

            return None, None

        # Load measurements
        with open(
            self.measurement_file,
            "r"
        ) as file:

            measurements = json.load(file)

        # Load evaluation
        with open(
            self.evaluation_file,
            "r"
        ) as file:

            evaluation = json.load(file)

        return measurements, evaluation

    def answer_question(self, question):

        print("\n===================================")
        print("       QUESTION / ANSWER AGENT")
        print("===================================")

        measurements, evaluation = self.load_data()

        if measurements is None:

            return "Required AI results are not available."

        prompt = f"""
You are the Question Answering Agent of an
AI-assisted Type A Aortic Dissection analysis system.

Answer the user's question using ONLY the actual
AI results provided below.

Do not invent information.

Do not change numerical values.

Do not make a definitive medical diagnosis.

Do not prescribe treatment.

If the question cannot be answered from the provided
data, clearly say that the information is not available.

ACTUAL MEASUREMENTS:

Segmented Voxels:
{measurements.get("segmented_voxels")}

Volume:
{measurements.get("volume_cm3")} cm3

Volume:
{measurements.get("volume_mm3")} mm3

Voxel Spacing:
{measurements.get("voxel_spacing_mm")}

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

USER QUESTION:

{question}

Provide a clear and concise answer.

If the question concerns clinical interpretation,
state that a qualified radiologist must perform
the final interpretation.
"""

        print("\nSending question to Llama 3.2...")

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        print("\n===================================")
        print("              ANSWER")
        print("===================================")

        print(answer)

        return answer


if __name__ == "__main__":

    agent = QuestionAgent()

    print("\nAI Question Answering Agent")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("User: ")

        if question.lower() == "exit":

            print("Question Agent stopped.")

            break

        if not question.strip():

            continue

        agent.answer_question(question)