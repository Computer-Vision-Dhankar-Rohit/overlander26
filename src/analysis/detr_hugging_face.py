#/analysis/detr_hugging_face.py

from transformers import (DetrImageProcessor, 
                            DetrForObjectDetection,
                            AutoImageProcessor,
                            AutoModelForObjectDetection)

from transformers import pipeline
checkpoint = "PekingU/rtdetr_v2_r50vd"
pipeline_rtdetr_v2 = pipeline("object-detection", model=checkpoint, image_processor=checkpoint)

#DetrFeatureExtractor) ##DetrFeatureExtractor ## Deprecated warning 
                            
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))
import torch , os , cv2 
import numpy as np
import supervision as sv
from PIL import Image
import requests ,math
import matplotlib.pyplot as plt ##Error--plot_results-

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

#%config InlineBackend.figure_format = 'retina'

# import ipywidgets as widgets
# from IPython.display import display, clear_output

# import torch
# from torch import nn
# from torchvision.models import resnet50
# import torchvision.transforms as T
# torch.set_grad_enabled(False);

# # you can specify the revision tag if you don't want the timm dependency
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# you can specify the revision tag if you don't want the timm dependency
detr_image_processor_detr_resnet101 = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
logger.warning("--init---detr_image_processor_detr_resnet101--> %s" ,type(detr_image_processor_detr_resnet101))
model_detr_resnet101 = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
logger.warning("--init---model_detr_resnet101---> %s" ,type(model_detr_resnet101))
#
detr_image_processor_detr_resnet101_dc5 = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101-dc5')
logger.warning("--init---detr_image_processor_detr_resnet101_dc5---> %s" ,type(detr_image_processor_detr_resnet101_dc5))
model_detr_resnet101_dc5 = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
logger.warning("--init---model_detr_resnet101_dc5---> %s" ,type(model_detr_resnet101_dc5))

# COCO classes
CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

from read_cam.read_webcam import CV2VideoCapture

class GetFramesFromVids:
    """ 
    """

    @classmethod
    def get_static_vids_local_list(self):
        """ 
        """
        try:
            init_vid_dir = CV2VideoCapture().get_init_dir()
            logger.debug("--get_static_vids_local_list---init_vid_dir-----> %s" ,init_vid_dir)
            
            ## TODO -- testing with Old Videos DIR 
            #init_vid_dir = "../data_dir/init_vid_dir_2025-02-02-21_08_/" ##init_vid_dir_2025-02-02-21_08_
            #init_vid_dir = "../data_dir/init_vid_small_1/" # OK 
            init_vid_dir = "../data_dir/init_vid_dir_2025-02-02-21_08_/"
            ls_video_files_uploads = []
            for root, dirs, files in os.walk(init_vid_dir):
                for filename in files:
                    ls_video_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of VIDEO-Files----> %s" ,ls_video_files_uploads)
            return ls_video_files_uploads
        except Exception as err:
            logger.error("-Error--get_static_vids_local_list---> %s" ,err)

    @classmethod
    def get_frame_from_video(self):
        """
        """
        print(f"-get_frame_from_video---HIT--> ")
        try:
            logger.debug("-get_frame_from_video---HIT->" )

            root_dir = "../data_dir/out_vid_frames_dir/"
            ls_frames_to_write = [4,11,17,25,30,37,45,55,66,77,88,100,110]
            ls_video_files_uploads = self.get_static_vids_local_list()
            for iter_vid in range(len(ls_video_files_uploads)):
                count = 0
                vid_short_name = str(ls_video_files_uploads[iter_vid])
                if "T" in str(vid_short_name):
                    vid_short_name = vid_short_name.rsplit("T",1)
                    print("---vid_short_name--a-TTT--\n",vid_short_name)
                    vid_short_name = str(str(vid_short_name[1]).rsplit("_",0))
                else:
                    pass
                if "mp4" in str(vid_short_name):
                    vid_short_name = vid_short_name.replace(".mp4","")
                else: # TODO - Video Format == #.webm
                    vid_short_name = vid_short_name.replace(".webm","")
                vid_short_name = vid_short_name.replace("['","")
                vid_short_name = vid_short_name.replace("']","")
                logger.debug("-Written--Video----> %s",vid_short_name)

                vidcap = cv2.VideoCapture(ls_video_files_uploads[iter_vid]) #eo_file__2025-01-26T11-32-45-853845_.mp4'
                count_of_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                logger.debug("-TOTAL--COUNT--Of--FRAMES---> %s",count_of_frames)

                for iter_frame in range(len(ls_frames_to_write)):
                    if ls_frames_to_write[iter_frame] >= count_of_frames -2:
                        pass 
                    else:

                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, ls_frames_to_write[iter_frame])
                        success, image = vidcap.read() #while success:
                        logger.debug("--WRITING-FRAME----type(image)--> %s",type(image)) ## Numpy nd.array 
                        # print('width of Image: ', image.shape[1]) ## Pixels -- width of Image:  1920
                        # print('height of Image:', image.shape[0]) ## Pixels 
                        print('Size of Image:', image.size) ## Pixels --- Size of Image: 6220800
                        if image.size >= 6000000: ##62,20,800 - MAX for JPEG -- 65,500
                            print('LArge-forJPEG-Size of Image:', image.size) ## Pixels --- Size of Image: 6220800
                            ## TODO -- Testing for JPEG Size Issues -- "__.jpg"

                            frame_save_path = root_dir + str(vid_short_name) + "_frame_"+ str(count)+"__.jpg" #"__.tif"
                        else:
                            frame_save_path = root_dir + str(vid_short_name) + "_frame_"+ str(count)+"__.jpg"

                        logger.debug("--WRITING-FRAME----> %s",frame_save_path)
                        #vidcap.release()
                        cv2.imwrite(frame_save_path, image) # save frame as .tif file      
                        count += 1
        except Exception as err:
            logger.error("-Error--get_frame_from_video---> %s" ,err)

class AnalysisVideoFrames:
    """ 
    """

    @classmethod
    def box_cxcywh_to_xyxy(self,x):     # for output bounding box post-processing
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @classmethod
    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

class PlotBboxOnFrames:
    """ 
    Desc:
        - Plot the Bounding Boxes on the Objects within each Image Frame
    """

    @classmethod
    def get_frames_local_list(self):
        """ 
        """
        try:
            root_dir = "../data_dir/out_vid_frames_dir/" #root_dir = "static/image_uploads/"
            ls_files_uploads = []
            for root, dirs, files in os.walk(root_dir):
                for filename in files:
                    ls_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of Image Files Uploaded ----> %s",ls_files_uploads)
            return ls_files_uploads
        except Exception as err:
            logger.error("-Error--get_frames_local_list---> %s" ,err)

    @classmethod
    def get_bbox_on_frames(self):
        """ 
        """
        try:
            ls_files_uploads = self.get_frames_local_list()
            for iter_img in range(len(ls_files_uploads)):
                image_name_for_zoom = ls_files_uploads[iter_img]
                image_name = str(ls_files_uploads[iter_img]).replace("../data_dir/out_vid_frames_dir/","")
                bbox_image_name = "bbox_"+ image_name
                logger.debug("--BBOX-IMAGE-NAME----> %s",bbox_image_name)
                image = Image.open(ls_files_uploads[iter_img])
                try:
                    inputs = detr_image_processor_detr_resnet101_dc5(images=image, return_tensors="pt") ## RESNET--101
                    outputs = model_detr_resnet101_dc5(**inputs)
                except Exception as err:
                    logger.error("-Error--detr_image_processor_detr_resnet101_dc5---> %s" ,err)

                probas = outputs.logits.softmax(-1)[0, :, :-1] ## ORIGINAL CODE 
                keep = probas.max(-1).values > 0.7 ## TODO -- keep any prediction with VALS 40% 
                # convert boxes from [0; 1] to image scales
                bboxes_scaled = AnalysisVideoFrames().rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
                self.plot_results(image, bbox_image_name,
                                    image_name_for_zoom,
                                    probas[keep], 
                                    bboxes_scaled)

                # convert outputs (bounding boxes and class logits) to COCO API
                # let's only keep detections with score > 0.9
                target_sizes = torch.tensor([image.size[::-1]])
                logger.debug("--target_sizes---> %s",target_sizes)
                results = detr_image_processor_detr_resnet101_dc5.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
                logger.debug("--ALL--Results---> %s",results)

                # logger.debug("--Lables within Results---> %s",results["labels"]) #print(results["labels"])
                # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                #     logger.debug("--BBOX--List within Results---> %s",box.tolist())
                #     logger.debug("--BBOX--[Label-Item] within Results---> %s",model_detr_resnet101_dc5.config.id2label[label.item()])
                #     BBox = [round(i, 2) for i in box.tolist()]
                #     logger.debug("--BBOX--[BBOX] within Results---> %s",BBox)
                #     # print(
                    #         f"Detected {model.config.id2label[label.item()]} with confidence "
                    #         f"{round(score.item(), 3)} at location {BBox}"
                    # )
                    #logger.debug("-Detected-BBOX-LABEL---> %s -List within Results---> %s",model.config.id2label[label.item()],box.tolist())
        except Exception as err:
            logger.error("-Error--get_bbox_on_frames---> %s" ,err)

    @classmethod
    def plot_results(self,
                     pil_img, 
                    bbox_image_name,
                    image_name_for_zoom,
                    prob, boxes):
        """ 
        Desc:
            - Method to Plot the Bounding Boxes on the Objects within each Image Frame
        """
        try:
            ls_car_images = []
            plt.figure(figsize=(25,25))
            plt.imshow(pil_img)
            ax = plt.gca()
            colors = COLORS * 100
            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3)) ## Drawing_BBox
                cl = p.argmax()
                text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                if "car" in str(text) or "person" in str(text) or "bottle" in str(text):
                    logger.debug("--BBOX-[CAR]-[PERSON]-[BOTTLE]--> %s",str(text))
                    ls_car_images.append(bbox_image_name)
                    # bbox_image_path = bbox_image_name 
                    #img_zoomed_and_cropped = self.zoom_center(image_name_for_zoom,zoom_factor=1.5)
                    root_dir = "../data_dir/zoomed_car_imgs/"
                    #cv2.imwrite(root_dir+bbox_image_name+'_zoomed.png', img_zoomed_and_cropped) # TODO -- save 
                    #cv2.imwrite(root_dir+bbox_image_name+'_zoomed.png', pil_img) # TODO -- save Original Not Zoomed 
                    ax.text(xmin, ymin, text, fontsize=15,bbox=dict(facecolor='yellow', alpha=0.5))
                    plt.axis('off')
                    root_save_dir = "../data_dir/bbox_image_saved/"
                    plt.savefig(root_save_dir+bbox_image_name, bbox_inches='tight') ## TODO -- save 
                    
                else:
                    pass ## Done save images without these BBOX -- 
            plt.close()
            return ls_car_images  
        except Exception as err:
            logger.error("-Error--plot_results---> %s" ,err)

    @classmethod
    def zoom_center(self,
                    img, 
                    zoom_factor=1.5):
        """ 
        get a zoomed image -- But centered
        """
        try:
            img = cv2.imread(img)
            y_size = img.shape[0]
            x_size = img.shape[1]
            x1 = int(0.5*x_size*(1-1/zoom_factor)) # define new boundaries
            x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
            y1 = int(0.5*y_size*(1-1/zoom_factor))
            y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))
            img_cropped = img[y1:y2,x1:x2] # first crop image then scale
            img_zoomed_and_cropped = cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
            return img_zoomed_and_cropped
        except Exception as err:
            logger.error("-Error--zoom_center---> %s" ,err)

class FaceDetection:
    """ 
    """

    @classmethod
    def invoke_model_yolov8_face_detection(self):
        """ 
        """
        # download model
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        logger.debug("----model_path--> %s" ,model_path)
        model_yolov8_face_detection = YOLO(model_path) # load model
        logger.debug("---model_yolov8_face_detection--> %s" ,type(model_yolov8_face_detection))
        return model_yolov8_face_detection

    @classmethod
    def face_detect_yolo_huggin_face(self,image_frame_path):
        """ 
        # TODO - SOURCE -- https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.RoundBoxAnnotator-functions
        """

        model_yolov8_face_detection = self.invoke_model_yolov8_face_detection()
        
        image_for_process = cv2.imread(image_frame_path) ## image_for_process = cv2.imdecode(Image.open(image_frame_path))
        output = model_yolov8_face_detection(Image.open(image_frame_path))
        results_face_detect = Detections.from_ultralytics(output[0]) #detections = sv.Detections(...)
        logger.debug("--results_face_detect---> %s" ,results_face_detect)
        logger.debug("--results_face_detect--Detections--aa-> %s" ,results_face_detect.xyxy)
        logger.debug("--results_face_detect--Detections--aa---TYPE--> %s" ,type(results_face_detect.xyxy)) #<class 'numpy.ndarray'>
        ls_faces_coord = results_face_detect.xyxy.tolist()
        for iter_face in range(len(ls_faces_coord)):  # Iterate through detected faces in current frame
            x1, y1, x2, y2 = ls_faces_coord[iter_face]  # Extract box coordinates
            logger.debug("--results_face_detect--x1, y1, x2, y2--> %s" ,x1)
            face_roi_region = image_for_process[int(y1):int(y2), int(x1):int(x2)]
            logger.debug("--results_face_detect----face_roi_region-> %s" ,face_roi_region)
            logger.debug("--results_face_detect----face_roi_region-> %s" ,type(face_roi_region))
            average_color_region = np.mean(face_roi_region, axis=(0, 1))  # Average over height and width
            logger.debug("--results_face_detect--TYPE----average_color_region-> %s" ,type(average_color_region))
            logger.debug("--results_face_detect----average_color_region-> %s" ,average_color_region)
            # Convert to integers (optional, but often useful for display/OpenCV)
            color_rgb_int = average_color_region.astype(int)
            # Print the integer RGB values
            print("Integer RGB values:", color_rgb_int)
            # If you need a tuple:
            color_rgb_tuple = tuple(color_rgb_int)
            print("RGB tuple:", color_rgb_tuple)
            # If you want to format it nicely:
            print(f"RGB: ({color_rgb_int[0]}, {color_rgb_int[1]}, {color_rgb_int[2]})")

            # --- Dominant Color Calculation (using k-means) --- k = 3
    
            k_means_k = 3
            pixels = face_roi_region.reshape(-1, 3).astype(np.float32)  # Reshape for k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k_means_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_colors = centers.astype(int)  # Convert back to integers
            logger.debug("--results_face_detect--TYPE----dominant_colors-> %s" ,type(dominant_colors))
            logger.debug("--results_face_detect---dominant_colors-> %s" ,dominant_colors)

            ## TODO -- Convert to COLOR NAMES 
            ## https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python

        # for result in results_face_detect: 
        #     boxes = result.boxes  # Get the bounding boxes
        #     xyxy = boxes.xyxy.astype(int)  # Convert box coordinates to integers (important for OpenCV)
        #     conf = boxes.conf  # Get confidence scores
        #     cls = boxes.cls  # Get class labels (usually 0 for face detection)

        image_face_detected = cv2.imread(image_frame_path)        

        box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN,thickness=5) ## TODO - color=blue -
        label_annotator = sv.LabelAnnotator(color=sv.Color.GREEN,
                                           text_color=sv.Color.BLACK,
                                           text_thickness=3,
                                           text_position=sv.Position.BOTTOM_RIGHT)
        
        # Create custom labels for each detection
        labels = ["FACE_ID_By_OVERLANDER"] * len(results_face_detect)
        
        annotated_frame = box_annotator.annotate(
                                scene=image_face_detected.copy(),
                                detections=results_face_detect)
        
        annotated_frame = label_annotator.annotate(
                                scene=annotated_frame,
                                detections=results_face_detect,
                                labels=labels)
        
        logger.debug("--results_face_detect-----annotated_frame---> %s" ,type(annotated_frame))

        logger.debug("--results_face_detect-----image_frame_path---> %s" ,image_frame_path)
        ##./data_dir/out_vid_frames_dir/21-08-21-070423__frame_6__.jpg
        print("--type-----image_frame_path---",type(image_frame_path))
        
        # Extract just the filename from the full path using os.path.basename
        image_face_detect = os.path.basename(image_frame_path)
        # Remove the file extension
        image_face_detect = os.path.splitext(image_face_detect)[0]
        
        print("-image_face_detect--",image_face_detect)
        logger.debug("--results_face_detect-----image_face_detect---> %s" ,image_face_detect)
        print("    "*90)
        
        # Define output directory with relative path
        face_out_rootDIR = "../data_dir/out_dir/"
        # Check if directory exists, create if not
        if not os.path.exists(face_out_rootDIR):
            os.makedirs(face_out_rootDIR)
            logger.debug("--Created output directory---> %s" ,face_out_rootDIR)
        
        cv2.imwrite(face_out_rootDIR+image_face_detect+'__face_.png', annotated_frame)
        logger.debug("--Saved face detection image to---> %s" ,face_out_rootDIR+image_face_detect+'__face_.png')

class ObjDetHFRtDetr:
    """
    """
    @classmethod
    def object_detect_RT_DETR(self,image_frame_path):
        """ 
        """
        image_local = Image.open(image_frame_path) #
        ls_results = pipeline_rtdetr_v2(image_frame_path, threshold=0.3)
        print("---type(res)---------",type(ls_results))
        print("   "*100)
        self.draw_PIL_Image(ls_results,
                            image_local,
                            image_frame_path)
        return ls_results
    
    @classmethod
    def draw_PIL_Image(self,ls_results,
                       image_local,
                       image_frame_path):
        """ 
        """
        from PIL import ImageDraw
        LS_COLORS = ["green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red"]
        # LS_COLORS = [
        #                 [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #                 [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
        #             ] * 100
        
        # colors for visualization
        # COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #         [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        annotated_image = image_local.copy()
        #draw = ImageDraw.Draw(annotated_image)
        plt.figure(figsize=(25,25))
        plt.imshow(annotated_image)
        ax = plt.gca()
        try:
            for iter_k in range(len(ls_results)): ## Original -- #for i, result in enumerate(ls_results):
                dict_res_1 = ls_results[iter_k]#["box"]
                print(dict_res_1)
                print("   "*100) ##{'score': 0.31572669744491577, 'label': 'cup', 'box': {'xmin': 407, 'ymin': 498, 'xmax': 440, 'ymax': 620}}
                dict_box = dict_res_1["box"]
                print(dict_box)
                #color = tuple([int(x * 255) for x in LS_COLORS[iter_k]])
                xmin, ymin, xmax, ymax = dict_box["xmin"], dict_box["ymin"], dict_box["xmax"], dict_box["ymax"]
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, color=LS_COLORS[iter_k], linewidth=4)) # Drawing_BBox
                ax.text(xmin, ymin,dict_res_1["label"],fontsize=25,
                        bbox=dict(facecolor='yellow', alpha=0.5)) # Drawing_LABEL
                plt.axis('off')
                
                # Define output directory with relative path
                root_save_dir = "../data_dir/out_dir/"
                # Check if directory exists, create if not
                if not os.path.exists(root_save_dir):
                    os.makedirs(root_save_dir)
                    logger.debug("--Created output directory---> %s" ,root_save_dir)
                
                # Extract just the filename using os.path.basename
                image_named_bbox = os.path.basename(image_frame_path)
                # Remove the file extension
                image_named_bbox = os.path.splitext(image_named_bbox)[0]
                print("---image_named_bbox----",image_named_bbox)
                print("- -OK-----   "*10)
                plt.savefig(root_save_dir+str(image_named_bbox)+".png", bbox_inches='tight')
                logger.debug("--Saved object detection image to---> %s" ,root_save_dir+str(image_named_bbox)+".png")
        except Exception as err:
            print(err)
            pass

        #     draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=color, width=7)
        #     draw.text((xmin, ymin, xmax, ymax), text=dict_res_1["label"],fill="yellow",align ="left")#font=10)
        # annotated_image.show()
        # annotated_image.save("anno_img_ImageDraw_1.png")


#------------#------------#------------#------------#------------#------------
# read original 
# img = cv2.imread('original.png')

# # call our function
# img_zoomed_and_cropped = zoom_center(img)

# # write zoomed and cropped version
# cv2.imwrite('zoomed_and_cropped.png', img_zoomed_and_cropped)


######---------------
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)



######-------# ORIGINAL CODE --------
# inputs = processor(images=image, return_tensors="pt") ## RESNET--101
# outputs = model(**inputs) ## RESNET--101

#print(f"---Model-outputs->> {outputs}")
#DetrObjectDetectionOutput.logits
#outputs.logits
#print(f"---Model-outputs->> {outputs.logits}") ## Original Code had -- pred_logits
# keep only predictions with 0.7+ confidence
#probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] ## ORIGINAL CODE 


######-------# FACIAL LANDMARKS DETECTION --------
class FacialLandmarksDetection:
    """
    Class for detecting and annotating facial keypoints/landmarks on detected faces.
    Uses dlib for reliable facial landmark detection (68 landmarks per face).
    """
    
    def __init__(self):
        """
        Initialize dlib face landmark detector.
        """
        try:
            import dlib
            # Download the predictor model if needed
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            
            if not os.path.exists(predictor_path):
                print("--Downloading dlib facial landmark model...")
                import urllib.request
                import bz2
                url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
                
                # Extract the .bz2 file
                with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as source:
                    with open(predictor_path, 'wb') as dest:
                        dest.write(source.read())
                print("--Model downloaded and extracted successfully--")
            
            self.predictor = dlib.shape_predictor(predictor_path)
            print("--dlib facial landmark detector initialized successfully--")
            logger.debug("--dlib facial landmark detector initialized successfully--")
        except Exception as err:
            logger.error("--Error initializing dlib---> %s", err)
            print(f"--ERROR: dlib initialization failed: {err}")
            print("--Please install dlib: pip install dlib")
            raise
    
    @classmethod
    def detect_landmarks_on_face(cls, image_frame_path, face_bbox_list):
        """
        Detect facial landmarks on each detected face region.
        
        Input Parameters:
        - image_frame_path (str): Path to the input image file
        - face_bbox_list (list): List of face bounding boxes in format [[x1, y1, x2, y2], ...]
        
        Processing:
        - Reads the image using cv2
        - Extracts each face ROI based on bounding box
        - Runs dlib shape predictor on each face ROI
        - Detects 68 facial landmarks per face
        - Returns landmark coordinates for each face
        
        Output Parameters:
        - Returns: dict with face_index as key and landmarks array as value
        """
        try:
            import dlib
            instance = cls()
            image = cv2.imread(image_frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            all_face_landmarks = {}
            
            for face_idx, bbox in enumerate(face_bbox_list):
                x1, y1, x2, y2 = map(int, bbox)
                
                print(f"--Processing face {face_idx} with bbox: [{x1}, {y1}, {x2}, {y2}]")
                
                # Create dlib rectangle from bbox
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                
                # Detect landmarks
                shape = instance.predictor(image_rgb, dlib_rect)
                
                # Convert to numpy array
                landmarks_coords = []
                for i in range(68):  # dlib detects 68 landmarks
                    x = shape.part(i).x
                    y = shape.part(i).y
                    landmarks_coords.append([x, y, 0])  # z=0 for 2D landmarks
                
                all_face_landmarks[face_idx] = np.array(landmarks_coords)
                print(f"--Detected {len(landmarks_coords)} landmarks for face {face_idx}")
                logger.debug("--Detected %d landmarks for face %d-->", len(landmarks_coords), face_idx)
            
            print(f"--Total faces processed: {len(all_face_landmarks)}")
            return all_face_landmarks
            
        except Exception as err:
            print(f"--ERROR in detect_landmarks_on_face: {err}")
            logger.error("--Error in detect_landmarks_on_face---> %s", err)
            return {}
    
    @classmethod
    def annotate_landmarks_on_image(cls, image_frame_path, face_landmarks_dict, output_path=None):
        """
        Annotate facial landmarks on the image.
        
        Input Parameters:
        - image_frame_path (str): Path to the input image file
        - face_landmarks_dict (dict): Dictionary with face_index as key and landmarks as value
        - output_path (str, optional): Path to save annotated image
        
        Processing:
        - Reads the original image
        - Draws circles for each landmark point
        - Draws connections between landmarks (mesh)
        - Optionally saves the annotated image
        
        Output Parameters:
        - Returns: annotated_image (numpy.ndarray)
        """
        try:
            instance = cls()
            image = cv2.imread(image_frame_path)
            annotated_image = image.copy()
            
            print(f"--Starting annotation for {len(face_landmarks_dict)} faces")
            
            for face_idx, landmarks in face_landmarks_dict.items():
                if landmarks is None:
                    print(f"--Skipping face {face_idx} - no landmarks detected")
                    continue
                
                print(f"--Annotating face {face_idx} with {len(landmarks)} landmarks")
                
                # Draw landmark points
                for landmark in landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)  # Green dots
                
                logger.debug("--Annotated %d landmarks for face %d-->", len(landmarks), face_idx)
            
            # Save if output path provided
            if output_path:
                print(f"--Attempting to save to: {output_path}")
                
                # Check if directory exists, create if not
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"--Created directory: {output_dir}")
                
                success = cv2.imwrite(output_path, annotated_image)
                if success:
                    print(f"--SUCCESS: Saved landmarks image to: {output_path}")
                    logger.debug("--Saved landmarks annotated image to---> %s", output_path)
                else:
                    print(f"--ERROR: Failed to save image to: {output_path}")
                    logger.error("--Failed to save landmarks image to---> %s", output_path)
            else:
                print("--No output path provided, skipping save")
            
            return annotated_image
            
        except Exception as err:
            logger.error("--Error in annotate_landmarks_on_image---> %s", err)
            return None
    
    @classmethod
    def detect_and_annotate_full_pipeline(cls, image_frame_path, face_bbox_list, output_dir="../data_dir/out_dir/"):
        """
        Complete pipeline: Detect landmarks and annotate on image.
        
        Input Parameters:
        - image_frame_path (str): Path to the input image file
        - face_bbox_list (list): List of face bounding boxes
        - output_dir (str): Directory to save output images
        
        Processing:
        - Detects facial landmarks on all faces
        - Annotates landmarks on the image
        - Saves annotated image with suffix '_landmarks.png'
        
        Output Parameters:
        - Returns: tuple (face_landmarks_dict, annotated_image)
        """
        try:
            # Detect landmarks
            face_landmarks_dict = cls.detect_landmarks_on_face(image_frame_path, face_bbox_list)
            
            # Prepare output path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            image_name = os.path.basename(image_frame_path)
            image_name_no_ext = os.path.splitext(image_name)[0]
            output_path = os.path.join(output_dir, f"{image_name_no_ext}_landmarks.png")
            
            # Annotate and save
            annotated_image = cls.annotate_landmarks_on_image(
                image_frame_path, 
                face_landmarks_dict, 
                output_path
            )
            
            logger.debug("--Completed facial landmarks pipeline for---> %s", image_frame_path)
            return face_landmarks_dict, annotated_image
            
        except Exception as err:
            logger.error("--Error in detect_and_annotate_full_pipeline---> %s", err)
            return {}, None
