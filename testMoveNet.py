# general
import argparse
import sys
import time

#audio processing
import sounddevice as sd

#computer vision
# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

#osc comunication
from pythonosc import udp_client
   

from aux_functions import build_pitch_grid


def compute_bbox(keypoints, threshold=0):
    try:
        valid_keypoints = tf.gather(keypoints[0][0], indices=tf.where(keypoints[0][0][:,2]>threshold).numpy(), axis=0)
        valid_keypoints = tf.reshape(valid_keypoints, [valid_keypoints.shape[0],valid_keypoints.shape[2]])
        max_x = tf.reduce_max(valid_keypoints[:,1]).numpy()
        max_y = tf.reduce_max(valid_keypoints[:,0]).numpy()
        min_x = tf.reduce_min(valid_keypoints[:,1]).numpy()
        min_y = tf.reduce_min(valid_keypoints[:,0]).numpy()

        return min_x, min_y, max_x-min_x, max_y-min_y
    except Exception as e:
        print(f"Exception in compute_bbox: {e}")

def compute_gravity_center(keypoints, threshold=0):
    try:
        valid_keypoints = tf.gather(keypoints[0][0], indices=tf.where(keypoints[0][0][:,2]>threshold).numpy(), axis=0)
        valid_keypoints = tf.reshape(valid_keypoints, [valid_keypoints.shape[0],valid_keypoints.shape[2]])
        x_coord =tf.reduce_mean(valid_keypoints[:,1]).numpy()
        y_coord =tf.reduce_mean(valid_keypoints[:,0]).numpy()

        return (x_coord, y_coord)
    except Exception as e:
        print(f"Exception in compute_gravity_center: {e}")



def load_moveNet_model():
    try:
        # Download the model from TF Hub.
        model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
        movenet = model.signatures['serving_default']
        return movenet
    except Exception as e:
        print(f"Exception in loading the model: {e}")



def draw_point(img,x_coord, y_coord, radius = 2, color = (0,180,0), thickness=2):
    try:
        y, x, _ = img.shape
        img = cv2.circle(img, (int(x_coord*x), int(y_coord*y)), radius, color, thickness)

        return img
    except Exception as e:
        print(f"Exception in draw_point: {e}")

def draw_keypoints(img, keypoints, radius = 2, color = (0,180,0), thickness=2, threshold=0):
    try:
        #y, x, _ = img.shape
        # iterate through keypoints
        for idx, k in enumerate(keypoints[0,0,:,:]):
            # Converts to numpy array
            k = k.numpy()
            # Checks confidence for keypoint
            if k[2] > threshold:
                # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
                #yc = int(k[0] * y)
                #xc = int(k[1] * x)

                img = draw_point(img=img, x_coord=k[1],y_coord= k[0], radius=radius, color=color, thickness=thickness)
                # Draws a circle on the image for each keypoint
                #img = cv2.circle(img, (xc, yc), radius, color, thickness)


        return img
    except Exception as e:
        print(f"Exception in draw_keypoints: {e}")


def draw_bbox(img, x_coord, y_coord, height, length, color = (180,0,0), thickness=2):
    try:
        y, x, _ = img.shape
        low_left = (int(x_coord*x), int(y_coord*y))
        low_right = (int((x_coord+length)*x), int(y_coord*y))
        up_left = (int(x_coord*x), int((y_coord+height)*y))
        up_right = (int((x_coord+length)*x), int((y_coord+height)*y))

        cv2.line(img, low_left, low_right, color, thickness) 
        cv2.line(img, low_left, up_left, color, thickness) 
        cv2.line(img, up_left, up_right, color, thickness) 
        cv2.line(img, low_right, up_right, color, thickness) 
        return img
    except Exception as e:
        print(f"Exception in draw_bbo: {e}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default="127.0.0.1",
            help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=57120,
            help="The port the OSC server is listening on")
        parser.add_argument(
            '-debug', nargs='?', metavar='DEBUG', type=bool, default=False,
            help='debugging (default: %(default)s)')
        args = parser.parse_args()

        OSC_client = udp_client.SimpleUDPClient(args.ip, args.port)
        movenet = load_moveNet_model()
        # Threshold for 
        threshold = .3

        # Loads video source (0 is for main webcam)
        video_source = 0
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS)  #get the FPS of the videos

        # Checks errors while opening the Video Capture
        if not cap.isOpened():
            print('Error loading video')
            quit()


        success, img = cap.read()

        if not success:
            print('Error reding frame')
            quit()


        y, x, _ = img.shape
        pitch_grid = build_pitch_grid(x_dim=x, y_dim=y, printGrid=True)
        while success:
            #print(img.shape)
            #time.sleep(10)
            # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
            tf_img = cv2.resize(img, (256,256))
            tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
            tf_img = np.asarray(tf_img)
            tf_img = np.expand_dims(tf_img,axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            image = tf.cast(tf_img, dtype=tf.int32)

            # Run model inference.
            outputs = movenet(image)
            if args.debug:
                print(outputs)
            #break

            # Output is a [1, 1, 17, 3] tensor.
            # [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
            #[x,y, confidence]
            keypoints = outputs['output_0']

            
            x_bbox, y_bbox, length_bbox,height_bbox = compute_bbox(keypoints, threshold=threshold)
            bbox_area = height_bbox*length_bbox
            grav_center = compute_gravity_center(keypoints, threshold=threshold)


            #img = draw_keypoints(img = img, keypoints=keypoints)
            #img = draw_bbox(img=img,x_coord=x_bbox, y_coord=y_bbox, height=height_bbox, length=length_bbox)
            #img = draw_point(img=img, x_coord=grav_center[0], y_coord=grav_center[1])

            #freq = keypoints[0][0][0][0].numpy()*200 +200
            
            
            y, x, _ = img.shape
            x_coord = min(round(keypoints[0][0][0][1].numpy()*x), x-1)
            y_coord = min(round(keypoints[0][0][0][0].numpy()*y),y-1)
            print(x_coord, y_coord, round(keypoints[0][0][0][1].numpy()*x), round(keypoints[0][0][0][0].numpy()*y))
            freq = pitch_grid[x_coord][y_coord]
            print('freq:', freq)
            #print(freq,keypoints[0][0][0][0].numpy(), "  |  ",  x_bbox, y_bbox, height_bbox, length_bbox, "  |  ", grav_center)
            #print(keypoints)
            OSC_client.send_message("/freq", freq)

            #time.sleep(10)
            # Shows image
            cv2.imshow('Movenet', img)
            # Waits for the next frame, checks if q was pressed to quit
            if cv2.waitKey(1) == ord("q"):
                break

            # Reads next frame
            success, img = cap.read()

        cap.release()


    except Exception as e:
        print(f"Exception in main: {e}")

