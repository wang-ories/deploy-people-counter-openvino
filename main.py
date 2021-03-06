
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    """
        Connect to the MQTT client
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


#
def frame_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            current_count = current_count + 1
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold
    request_id = 0
    total_count = 0
    last_count = 0
    current_count = 0

    single_image_mode = False

    # Load the model through `infer_network`
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, request_id, args.cpu_extension)[1]

    # Handle the input stream

    # Check for image input
    if args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    elif args.input == 'CAM':
        input_stream = 0

    else: # input is a video path
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    global initial_w, initial_h
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Loop until stream is over

    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()

        if not flag:
            break

        key_pressed = cv2.waitKey(60)
        # Pre-process the image as needed
        image = cv2.resize(frame, (w, h))
        # Change data layout form HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        # Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(request_id, image)
        # Wait for the result
        if infer_network.wait(request_id) == 0:
            det_time = time.time() - inf_start
            # Get the results of the inference request
            result = infer_network.get_output(request_id)

            inf_time_message = "Inference time: {:.3f}ms" \
                .format(det_time * 1000)

            # Extract any desired stats from the results
            frame, current_count = frame_out(frame, result)
            cv2.putText(frame, "Current count number {} ".format(current_count),
                        (20, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, inf_time_message, (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)


            # Calculate and send relevant information on
            if current_count > last_count:
                start_time = time.time()
                # current_count, total_count and duration to the MQTT server
                total_count = total_count + current_count - last_count
                # Topic "person": keys of "count" and "total" ###
                client.publish("person", json.dumps({"total": total_count}))


            # Topic "person/duration": key of "duration"
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server ###

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    exit(0)

# model link :
# 1 - resources/FP32/mobilenet-ssd.xml

#/opt/intel/openvino/deployment_tools/model_optimizer# python3 mo_tf.py  --input_model /home/workspace/resources/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb  --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/resources/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels  -o /home/workspace/resources/

#python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/resources/frozen_inference_graph.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

#-l --block-size=MB
