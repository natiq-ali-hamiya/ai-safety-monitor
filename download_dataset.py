from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # paste your real key here

# Dataset 1 - Manholes
project = rf.workspace("hazard-road-detection").project("manhole-v6vat")
dataset = project.version(1).download("yolov8", location="D:/datasets/manhole")

# Dataset 2 - Construction Hazard
project2 = rf.workspace("object-detection-qn97p").project("construction-hazard-detection")
dataset2 = project2.version(43).download("yolov8", location="D:/datasets/construction-hazard")

# Dataset 3 - Construction Site Safety
project3 = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
dataset3 = project3.version(1).download("yolov8", location="D:/datasets/construction-safety")

print("All datasets downloaded to D drive!")