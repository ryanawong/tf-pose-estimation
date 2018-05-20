import argparse
import logging
import time

import cv2
import numpy as np
import common

from common import CocoPairsNetwork, CocoPairs, CocoPart
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
prev_time = 0
num_img = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--output', type=str, default='../collected_data/', help='input relative path to output images/csv')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # Start a file to write joint position values
    f = open("%spose_estimation_%f.csv" % (args.output, time.time()), "w+")
    f.write("Time,NoseX,NoseY,NeckX,NeckY,RShoulderX,RShoulderY,RElbowX,RElbowY,RWristX,RWristY,LShoulderX,LShoulderY,LElbowX,LElbowY,LWristX,LWristY,RHipX,RHipY,RKneeX,RKneeY,RAnkleX,RAnkleY,LHipX,LHipY,LKneeX,LKneeY,LAnkleX,LAnkleY,REyeX,REyeY,LEyeX,LEyeY,REarX,REarY,LEarX,LEarY\n")

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
	img_h, img_w = image.shape[:2]

	#################################
	### Beginning of showing joint coordinates and calculating knee bends
	#################################

	# 0 = nose; 1 = bottom of neck; 2 = right shoulder; 3 = right elbow; 4 = right wrist; 5 = left shoulder; 6 = left elbow; 7 = left wrist; 8 = right hip; 9 = right knee
	# 10 = right ankle; 11 = left hip; 12 = left knee; 13 = left ankle; 14 = right eye; 15 = left eye; 16 = right ear; 17 = left ear
	for human in humans:
	    if 11 in human.body_parts.keys() and 12 in human.body_parts.keys():
		left_hip = human.body_parts[11]
		left_knee = human.body_parts[12]
                center_left_hip = (int(left_hip.x * img_w + 0.5), int(left_hip.y * img_h + 0.5))
                center_left_knee = (int(left_knee.x * img_w + 0.5), int(left_knee.y * img_h + 0.5))
                cv2.putText(image, "Left Hip (%d, %d)" % center_left_hip, center_left_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, common.CocoColors[11], 2)
                cv2.putText(image, "Left Knee (%d, %d)" % center_left_knee, center_left_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.5, common.CocoColors[12], 2)

	    if 8 in human.body_parts.keys() and 9 in human.body_parts.keys():
                right_hip = human.body_parts[8]
                right_knee = human.body_parts[9]
                center_right_hip = (int(right_hip.x * img_w + 0.5), int(right_hip.y * img_h + 0.5))
                center_right_knee = (int(right_knee.x * img_w + 0.5), int(right_knee.y * img_h + 0.5))
                cv2.putText(image, "Right Hip (%d, %d)" % center_right_hip, center_right_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, common.CocoColors[8], 2)
                cv2.putText(image, "Right Knee (%d, %d)" % center_right_knee, center_right_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.5, common.CocoColors[9], 2)

        #################################
        ### Calculate knee bends based on angle between right hip and right knee
        #################################

	    if 8 in human.body_parts.keys() and 9 in human.body_parts.keys(): # and 1 in human.body_parts.keys() and 11 not in human.body_parts.keys() and 12 not in human.body_parts.keys():
                neck = human.body_parts[1]
		right_hip = human.body_parts[8]
                right_knee = human.body_parts[9]
                center_neck = (int(neck.x * img_w + 0.5), int(right_hip.y * img_h + 0.5))
		center_right_hip = (int(right_hip.x * img_w + 0.5), int(right_hip.y * img_h + 0.5))
                center_right_knee = (int(right_knee.x * img_w + 0.5), int(right_knee.y * img_h + 0.5))
#		AB = (center_right_hip[0] - center_neck[0], center_right_hip[1] - center_neck[1])
#		BC = (center_right_knee[0] - center_right_hip[0], center_right_knee[1] - center_right_hip[0])
                AB = (center_right_knee[0] - center_right_hip[0], center_right_knee[1] - center_right_hip[1])
                BC = (center_right_knee[0] - center_right_hip[0], center_right_knee[1] - center_right_knee[1])

                print(np.linalg.norm(AB), np.linalg.norm(BC), np.dot(AB, BC))
		
        #################################
        ### Only take a picture if we identify right knee and right hip in the frame
        #################################

		if AB != (0,0) and BC != (0,0):
		    angle = np.arccos(np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC)))
   		    if angle > 0.70 and angle < 0.80:
		    	cv2.putText(image, "Good Angle %.2f" % angle, (10, img_h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
			if time.time() - prev_time > 2:
			    cv2.imwrite("%simage%d_%f.jpg" % (args.output, num_img, time.time()), image)
			    num_img += 1
			    prev_time = time.time()
		    else:
		    	cv2.putText(image, "Bad Angle %.2f %.2f" % (angle, time.time() - prev_time), (10, img_h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        if time.time() - prev_time > 2:
                            cv2.imwrite("%simage%d_%f.jpg" % (args.output, num_img, time.time()), image)
			    num_img += 1
                            prev_time = time.time()


#	    else if 11 in human.body_parts.keys() and 12 in human.body_parts.keys() and 1 in human.body_parts.keys() and 8 not in human.body_parts.keys() and 9 not in human.body_parts.keys():
#                cv2.putText(img, "Bad Angle %d" % angle, (10, img_h), cv2.FONT_HERSHET_SIMPLEX, 0.5, (255, 0, 0), 2)
	    f.write("%f" % time.time())
	    for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
		    f.write(",000,000")
		    continue
		first = 0
                body_part = human.body_parts[i]
                center = (int(body_part.x * img_w + 0.5), int(body_part.y * img_h + 0.5))
		f.write(",%3d,%3d" % center);
#                cv2.putText(image, "%d" % i, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, common.CocoColors[i], 2)
	    f.write("\n")

	    #################################
	    ### End of showing joint coordinates and calculating knee bends
	    #################################

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27: # Press Esc to exit program
            break
        logger.debug('finished+')

    f.close()
    cv2.destroyAllWindows()
