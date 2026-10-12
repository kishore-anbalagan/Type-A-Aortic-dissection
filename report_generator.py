from datetime import datetime

dice = 0.82
iou = 0.71

report = f"""
--------------------------------------------
        AI RADIOLOGY REPORT
--------------------------------------------

Date : {datetime.now().strftime('%d-%m-%Y')}

Patient ID : TAAD_001

Study :
CT Angiography

AI Model :
3D U-Net

Segmentation Status :
Completed Successfully

Evaluation

Dice Score : {dice:.4f}

IoU Score : {iou:.4f}

Findings

The AI model identified a region consistent with
Type A Aortic Dissection involving the ascending aorta.

Recommendation

Immediate clinical correlation is advised.
CT angiography review by a radiologist is recommended.

--------------------------------------------
"""

print(report)

with open("reports/AI_Report.txt", "w") as f:
    f.write(report)

print("AI Report Saved Successfully!")