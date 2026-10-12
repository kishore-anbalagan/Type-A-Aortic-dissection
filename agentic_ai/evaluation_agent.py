import os
import json


class EvaluationAgent:

    def __init__(self):

        self.evaluation_file = "outputs/evaluation.json"

    def get_evaluation(self):

        print("\n===================================")
        print("        EVALUATION AGENT")
        print("===================================")

        if not os.path.exists(self.evaluation_file):

            print("Error: evaluation.json not found!")

            return None

        with open(
            self.evaluation_file,
            "r"
        ) as file:

            evaluation = json.load(file)

        dice = evaluation.get("dice_score")
        iou = evaluation.get("iou_score")

        print("Evaluation results loaded.")

        print("Dice Score :", dice)
        print("IoU Score  :", iou)

        return evaluation


if __name__ == "__main__":

    agent = EvaluationAgent()

    agent.get_evaluation()