import os
#from util_logger import setup_logger
# logger = setup_logger(module_name=str(__name__))
# from read_cam.read_webcam import CV2VideoCapture
from analysis.detr_hugging_face import (GetFramesFromVids , 
                                        PlotBboxOnFrames,
                                        FacialLandmarksDetection,
                                        FaceDetection,
                                        ObjDetHFRtDetr) #,PlotBboxOnFrames

#from analysis.hugging_face_rtdetr_v2 import AutoModelRtDetrV2

class IPWebCam:
    """ 
    """

    @classmethod
    def get_frames_local_list(self,root_dir):
        """ 
        """
        try:
            #root_dir = "../data_dir/out_vid_frames_dir/" #root_dir = "static/image_uploads/"
            ls_files_uploads = []
            for root, dirs, files in os.walk(root_dir):
                for filename in files:
                    ls_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of Image Files Uploaded ----> %s",ls_files_uploads)
            return ls_files_uploads
        except Exception as err:
            logger.error("-Error--get_frames_local_list---> %s" ,err)

    @classmethod
    def invoke_scan(self):
        """`
        """
        logger.debug(f"-invoke_scan--hit-> ")
        CV2VideoCapture().video_cap_init()

    @classmethod
    def analyse_scan(self):
        """`
        """
        GetFramesFromVids().get_frame_from_video()
        PlotBboxOnFrames().get_bbox_on_frames()

    @classmethod
    def face_detect_yolo_hface(self):
        """ 
        """
        image_rootDIR="/home/dhankar/temp/09_25/off_1/jungle_images/deepface_sample_images"
        ls_files_uploads = self.get_frames_local_list(image_rootDIR)
        for iter_k in range(len(ls_files_uploads)):
            image_local_path = ls_files_uploads[iter_k]
            print("--image_local_path----",image_local_path)
            FaceDetection().face_detect_yolo_huggin_face(image_local_path)

    @classmethod
    def face_detect_with_landmarks(self):
        """
        End-to-end pipeline for face detection and facial landmarks detection.
        
        Processing:
        - Detects faces using YOLO
        - Extracts face bounding boxes
        - Runs facial landmarks detection on each face
        - Saves annotated images with landmarks
        """
        try:
            print(f"-face_detect_with_landmarks--hit-> ")
            logger.debug(f"-face_detect_with_landmarks--hit-> ")
            
            # Set image directory
            image_rootDIR = "/home/dhankar/temp/09_25/off_1/jungle_images/deepface_sample_images"
            ls_files_uploads = self.get_frames_local_list(image_rootDIR)
            
            for iter_k in range(len(ls_files_uploads)):
                image_local_path = ls_files_uploads[iter_k]
                print("--Processing image for landmarks----", image_local_path)
                logger.debug("--Processing image for landmarks----> %s", image_local_path)
                
                # Step 1: Detect faces using YOLO
                face_detection_instance = FaceDetection()
                model_yolov8 = face_detection_instance.invoke_model_yolov8_face_detection()
                
                from PIL import Image
                from supervision import Detections
                
                output = model_yolov8(Image.open(image_local_path))
                results_face_detect = Detections.from_ultralytics(output[0])
                
                # Extract face bounding boxes
                face_bbox_list = results_face_detect.xyxy.tolist()
                logger.debug("--Detected %d faces for landmarks----> %s", len(face_bbox_list), image_local_path)
                
                if len(face_bbox_list) > 0:
                    # Step 2: Run facial landmarks detection pipeline
                    face_landmarks_dict, annotated_image = FacialLandmarksDetection.detect_and_annotate_full_pipeline(
                        image_local_path,
                        face_bbox_list,
                        output_dir="../data_dir/out_dir/"
                    )
                    
                    print(f"--Completed landmarks detection for {len(face_bbox_list)} faces")
                    logger.debug("--Completed landmarks for---> %s with %d faces", image_local_path, len(face_bbox_list))
                else:
                    print(f"--No faces detected in image: {image_local_path}")
                    logger.warning("--No faces detected in image---> %s", image_local_path)
                    
        except Exception as err:
            logger.error("--Error in face_detect_with_landmarks---> %s", err)
            print(f"Error in face_detect_with_landmarks: {err}")

    @classmethod
    def object_detect_HFRtDetr_pipeline(self):
        """ 
        Desc:
            - pipeline processed - Not direct Model 
        """
        try:
            image_frame_path = "../data_dir/jungle_images/input_DIR/"
            ls_files_uploads = self.get_frames_local_list(image_frame_path)
            for iter_k in range(len(ls_files_uploads)):
                image_frame = ls_files_uploads[iter_k]
                print("--IMAGE--FRAME-----",image_frame)
                print("   ==FRA------   "*20)
                ObjDetHFRtDetr().object_detect_RT_DETR(image_frame)
        except Exception as err:
            print(err)

    @classmethod
    def object_detect_HFRtDetr_model(self):
        """ 
        Desc:
            - pipeline processed - Not direct Model 
        """
        try:
            image_rootDIR= "../data_dir/jungle_images/input_DIR/"
            ls_files_uploads = self.get_frames_local_list(image_rootDIR)
            for iter_k in range(len(ls_files_uploads)):
                image_local_path = ls_files_uploads[iter_k]
                print("--image_local_path----",image_local_path)
                image_detections , image_local_frame = AutoModelRtDetrV2().obj_detect_HFRtDetr_v2_model(image_local_path)
                logger.debug("--main.py--model_obj_detection--image_detections----aa---> %s" ,image_detections)
                AutoModelRtDetrV2().plot_results_HFRtDetr_v2_model(image_detections , image_local_frame,image_local_path)
        except Exception as err:
            logger.error("--main.py--object_detect_HFRtDetr_model-> %s" ,err)

if __name__ == "__main__":
    #IPWebCam().invoke_scan() #TODO -ARGPARSE required for main method calls
    #IPWebCam().analyse_scan()
    #IPWebCam().face_detect_yolo_hface()
    
    # INVOKE End-to-end Face Detection with Facial Landmarks
    IPWebCam().face_detect_with_landmarks()
    
    #IPWebCam().object_detect_HFRtDetr_pipeline()
    #IPWebCam().object_detect_HFRtDetr_model()
    

