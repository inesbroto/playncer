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
   




parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=57120,
    help="The port the OSC server is listening on")
parser.add_argument(
    '-debug', nargs='?', metavar='DEBUG', type=bool, default=False,
    help='debugging (default: %(default)s)')
args = parser.parse_args()


client = udp_client.SimpleUDPClient(args.ip, args.port)



# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

# Threshold for 
threshold = .2

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

skipping_samples = 29
curr_sample =0
time_of_frame = time.time()

while success:
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
    # iterate through keypoints

    for idx, k in enumerate(keypoints[0,0,:,:]):
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (xc, yc), 2, (0, 180, 0), 5)
            if args.debug:
                print(yc,xc,fps)
        
        if idx==0: freq = k[1]*200 +200

    #if curr_sample == 0: 
    #    print(freq,time.time()-time_of_frame)
    #    client.send_message("/freq", freq)
#
    #    time_of_frame = time.time()
    
    print(freq)
    client.send_message("/freq", freq)

    #curr_sample = (curr_sample + 1) % skipping_samples
    #
            #try:
            #    samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
            #    with sd.OutputStream(device=args.device, channels=1, callback=callback,
            #                        samplerate=samplerate):
            #        print('#' * 80)
            #        print('press Return to quit')
            #        print('#' * 80)
            #        input()
            #        #time.sleep(2)
#
            #except KeyboardInterrupt:
            #    parser.exit('')
            #except Exception as e:
            #    parser.exit(type(e).__name__ + ': ' + str(e))
#
            #if args.debug:
            #    print("Loop can continue!")


            #duration = 0.2  # seconds
            #sample_rate = 44100
            #t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            #frequency = xc  # A4 note
            #audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            #sd.play(audio, sample_rate)
            #sd.wait()

    # Shows image
    cv2.imshow('Movenet', img)
    # Waits for the next frame, checks if q was pressed to quit
    if cv2.waitKey(1) == ord("q"):
        break

    # Reads next frame
    success, img = cap.read()

cap.release()
