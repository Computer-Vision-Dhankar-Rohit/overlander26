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
    def face_detect_and_landmarks_combined(self):
        """
        Combined pipeline: Runs face detection and landmark detection, then concatenates both images side-by-side.
        LEFT: Face detection with green boxes and white "FACE_ID_By_OVERLANDER" labels
        RIGHT: Landmark detection with red dots
        """
        try:
            import cv2
            import numpy as np
            from analysis.detr_hugging_face import FaceDetection, FacialLandmarksDetection
            
            print("--Starting combined face detection and landmarks pipeline--")
            logger.info("--Starting combined pipeline--")
            
            # Get image list
            image_rootDIR = "/home/dhankar/temp/09_25/off_1/jungle_images/deepface_sample_images"
            image_list_path = self.get_frames_local_list(image_rootDIR)
            
            output_dir = "../data_dir/out_dir/"
            os.makedirs(output_dir, exist_ok=True)
            
            for image_local_path in image_list_path:
                print(f"\n--Processing image: {image_local_path}")
                
                # Step 1: Face Detection
                print("--Running face detection...")
                results_face_detect, face_bbox_list = FaceDetection.face_detect_yolo_huggin_face(image_local_path)
                
                if len(face_bbox_list) == 0:
                    print(f"--No faces detected in {image_local_path}, skipping...")
                    continue
                
                # Save face detection image to temp location
                face_detect_temp = os.path.join(output_dir, "temp_face_detect.png")
                annotated_face = FaceDetection.annotate_face_detection(image_local_path, results_face_detect)
                cv2.imwrite(face_detect_temp, annotated_face)
                print(f"--Face detection complete: {len(face_bbox_list)} faces detected")
                
                # Step 2: Landmark Detection
                print("--Running landmark detection...")
                face_landmarks_dict, annotated_landmarks = FacialLandmarksDetection.detect_and_annotate_full_pipeline(
                    image_local_path,
                    face_bbox_list,
                    output_dir=None  # Don't save yet
                )
                print(f"--Landmark detection complete")
                
                # Step 3: Load both images
                img_face_detect = cv2.imread(face_detect_temp)
                img_landmarks = annotated_landmarks
                
                # Step 4: Ensure both images have same height for concatenation
                h1, w1 = img_face_detect.shape[:2]
                h2, w2 = img_landmarks.shape[:2]
                
                if h1 != h2:
                    # Resize to match the smaller height
                    target_height = min(h1, h2)
                    img_face_detect = cv2.resize(img_face_detect, (int(w1 * target_height / h1), target_height))
                    img_landmarks = cv2.resize(img_landmarks, (int(w2 * target_height / h2), target_height))
                
                # Step 5: Concatenate horizontally (side by side)
                combined_image = np.hstack([img_face_detect, img_landmarks])
                
                # Step 6: Save combined image
                base_name = os.path.splitext(os.path.basename(image_local_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_combined.png")
                cv2.imwrite(output_path, combined_image)
                
                print(f"--SUCCESS: Saved combined image to {output_path}")
                logger.info("--Saved combined image---> %s", output_path)
                
                # Clean up temp file
                if os.path.exists(face_detect_temp):
                    os.remove(face_detect_temp)
            
            print("\n--Combined pipeline completed for all images--")
            
        except Exception as err:
            logger.error("--Error in face_detect_and_landmarks_combined---> %s", err)
            print(f"Error in combined pipeline: {err}")

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

    # TODO - INVOKE End-to-end Face Detection - and Facial Landmarks 
    # Combine the Output images - Creates side-by-side concatenated images
    IPWebCam().face_detect_and_landmarks_combined()
    
    # Old separate methods (use only if you want individual outputs):
    # IPWebCam().face_detect_yolo_hface()
    # IPWebCam().face_detect_with_landmarks()
    
    #IPWebCam().object_detect_HFRtDetr_pipeline()
    #IPWebCam().object_detect_HFRtDetr_model()
    

