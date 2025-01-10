from Detector import *
# Initialize the detector
detector = Detector(model_type="PS")

image_path = "image/h4.jpg"  # Path to your image
detector.onImage(image_path)
# detector.onImage(image_path, target_classes=["traffic light"])
# detector.onImage(image_path, target_classes=["person"])
detector.onImage(image_path, target_classes=["backpack"])
detector.onImage(image_path, target_classes=["handbag"])
# detector.onImage(image_path, target_classes=["wine glass"])
# detector.onImage(image_path, target_classes=["banner"])
detector.onVideo("C:/Users/battl/Downloads/cv/object_detection/image/t1.mp4") #path of the video