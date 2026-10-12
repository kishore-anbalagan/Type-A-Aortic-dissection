from measurement_agent import MeasurementAgent
from evaluation_agent import EvaluationAgent
from llm_agent import LLMAgent
from recommendation_agent import RecommendationAgent
from question_agent import QuestionAgent

import subprocess
import sys


class AgentController:

    def __init__(self):

        print("===================================")
        print("      AGENTIC AI CONTROLLER")
        print("===================================")

        self.measurement_agent = MeasurementAgent()
        self.evaluation_agent = EvaluationAgent()
        self.llm_agent = LLMAgent()
        self.recommendation_agent = RecommendationAgent()
        self.question_agent = QuestionAgent()

    # =========================================
    # AI ANALYSIS
    # =========================================

    def run_analysis(self):

        print("\nStarting AI Analysis Workflow...\n")

        # 1. Measurement Agent
        measurements = (
            self.measurement_agent.get_measurements()
        )

        if measurements is None:
            print("Measurement Agent failed.")
            return

        # 2. Evaluation Agent
        evaluation = (
            self.evaluation_agent.get_evaluation()
        )

        if evaluation is None:
            print("Evaluation Agent failed.")
            return

        # 3. Recommendation Agent
        recommendation = (
            self.recommendation_agent
            .generate_recommendation(
                measurements,
                evaluation
            )
        )

        # 4. LLM Agent
        report = self.llm_agent.generate_report(
            measurements,
            evaluation
        )

        # 5. Save report
        output_file = "outputs/agentic_report.txt"

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)

            file.write("\n\n")
            file.write(
                "===================================\n"
            )

            file.write(
                "AI DECISION SUPPORT RECOMMENDATION\n"
            )

            file.write(
                "===================================\n\n"
            )

            file.write(recommendation)

        print("\n===================================")
        print("      AI ANALYSIS COMPLETED")
        print("===================================")

        print(
            f"\nReport saved to: {output_file}"
        )

    # =========================================
    # QUESTION ANSWERING
    # =========================================

    def ask_question(self):

        print("\n===================================")
        print("       AI QUESTION ANSWERING")
        print("===================================")

        print(
            "Type 'exit' to return to the menu.\n"
        )

        while True:

            question = input("User: ")

            if question.lower() == "exit":
                break

            if not question.strip():
                continue

            self.question_agent.answer_question(
                question
            )

    # =========================================
    # GRAD-CAM
    # =========================================

    def run_gradcam(self):

        print("\n===================================")
        print("        GRAD-CAM AGENT")
        print("===================================")

        print(
            "\nGenerating Grad-CAM visualization...\n"
        )

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "gradcam.py"
                ],
                check=True
            )

            print("\n===================================")
            print("       GRAD-CAM COMPLETED")
            print("===================================")

            print(
                "\nGrad-CAM files are available in:"
            )

            print(
                "outputs\\gradcam"
            )

        except subprocess.CalledProcessError:

            print("\nGrad-CAM execution failed.")

        except FileNotFoundError:

            print(
                "\nError: gradcam.py not found."
            )

    # =========================================
    # MODEL COMPARISON
    # =========================================

    def compare_models(self):

        print("\n===================================")
        print("       MODEL COMPARISON")
        print("===================================")

        print(
            "\nComparing 3D U-Net, V-Net and SegResNet...\n"
        )

        try:

            subprocess.run(
                [
                    sys.executable,
                    "comparison\\compare_models.py"
                ],
                check=True
            )

            print("\n===================================")
            print("    MODEL COMPARISON COMPLETED")
            print("===================================")

        except subprocess.CalledProcessError:

            print(
                "\nModel comparison execution failed."
            )

        except FileNotFoundError:

            print(
                "\nError: comparison\\compare_models.py "
                "not found."
            )

    # =========================================
    # MAIN MENU
    # =========================================

    def run(self):

        while True:

            print("\n===================================")
            print("          AGENTIC AI SYSTEM")
            print("===================================")

            print(
                "1. Generate AI Radiology Report"
            )

            print(
                "2. Ask Question About Results"
            )

            print(
                "3. Generate Grad-CAM Visualization"
            )

            print(
                "4. Compare Segmentation Models"
            )

            print(
                "5. Exit"
            )

            choice = input(
                "\nSelect option: "
            )

            # Option 1
            if choice == "1":

                self.run_analysis()

            # Option 2
            elif choice == "2":

                self.ask_question()

            # Option 3
            elif choice == "3":

                self.run_gradcam()

            # Option 4
            elif choice == "4":

                self.compare_models()

            # Option 5
            elif choice == "5":

                print(
                    "\nAgentic AI System stopped."
                )

                break

            else:

                print(
                    "\nInvalid option. "
                    "Please select 1, 2, 3, 4 or 5."
                )


# =========================================
# PROGRAM START
# =========================================

if __name__ == "__main__":

    controller = AgentController()

    controller.run()