from roboflow import Roboflow
rf = Roboflow(api_key=<YOUR_API_KEY>)
project = rf.workspace("valorant-object-detection").project("valorant-object-detection-r9qkl")
version = project.version(22)
dataset = version.download("yolov8")

