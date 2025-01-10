from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2


class Detector:
    def __init__(self, model_type='OD'):

        self.cfg = get_cfg()
        self.model_type = model_type

        # Load the appropriate model configuration and weights
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        # elif model_type == "IS":
        #     self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #     self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # elif model_type == "KP":
        #     self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        #     self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        # elif model_type == "LVIS":
        #     self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_FPN_1x.yaml"))
        #     self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_FPN_1x.yaml")
        elif model_type == "PS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" 

        # Initialize the predictor
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath, target_classes=None):
        
        image = cv2.imread(imagePath)
        if image is None:
            print("Error: Unable to load image.")
            return

        # Get predictions based on the model type
        if self.model_type != "PS":
            predictions = self.predictor(image)
            instances = predictions["instances"].to("cpu")

            # Filter instances based on target classes
            if target_classes:
                class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
                target_indices = [i for i, label in enumerate(instances.pred_classes) if class_names[label] in target_classes]
                filtered_instances = instances[target_indices]
            else:
                filtered_instances = instances

            # Visualize the filtered instances
            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                             instance_mode=ColorMode.IMAGE_BW)
            output = viz.draw_instance_predictions(filtered_instances)
        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

            print("Available thing_classes:", metadata.thing_classes)
            print("Available stuff_classes:", metadata.stuff_classes)

            # Filter segments based on target classes
            if target_classes:
                filtered_segments = []
                for segment in segmentInfo:
                    category_id = segment["category_id"]
                    if category_id < len(metadata.thing_classes):
                        class_name = metadata.thing_classes[category_id]  # Thing classes
                    else:
                        class_name = metadata.stuff_classes[category_id - len(metadata.thing_classes)]  # Stuff classes

                    print(f"Segment class: {class_name}, Target: {target_classes}")

                    if class_name in target_classes:
                        filtered_segments.append(segment)
                segmentInfo = filtered_segments

                print("Filtered segmentInfo:", segmentInfo)

            # Visualize the panoptic segmentation results
            viz = Visualizer(image[:, :, ::-1], metadata=metadata)
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        # Display the result
        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath, target_classes=None):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        success, image = cap.read()

        while success:
            if self.model_type != "PS":
                predictions = self.predictor(image)
                instances = predictions["instances"].to("cpu")

                # Filter instances based on target classes
                if target_classes:
                    class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
                    target_indices = [i for i, label in enumerate(instances.pred_classes) if class_names[label] in target_classes]
                    filtered_instances = instances[target_indices]
                else:
                    filtered_instances = instances

                # Visualize the filtered instances
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.IMAGE_BW)
                output = viz.draw_instance_predictions(filtered_instances)
            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

                print("Available thing_classes:", metadata.thing_classes)
                print("Available stuff_classes:", metadata.stuff_classes)

                # Filter segments based on target classes
                if target_classes:
                    filtered_segments = []
                    for segment in segmentInfo:
                        category_id = segment["category_id"]
                        if category_id < len(metadata.thing_classes):
                            class_name = metadata.thing_classes[category_id]  # Thing classes
                        else:
                            class_name = metadata.stuff_classes[category_id - len(metadata.thing_classes)]  # Stuff classes

                        print(f"Segment class: {class_name}, Target: {target_classes}")

                        if class_name in target_classes:
                            filtered_segments.append(segment)
                    segmentInfo = filtered_segments

                    print("Filtered segmentInfo:", segmentInfo)

                # Visualize the panoptic segmentation results
                viz = Visualizer(image[:, :, ::-1], metadata=metadata)
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            # Display the result
            cv2.imshow("Result", output.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()