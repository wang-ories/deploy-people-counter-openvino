# Project Write-Up

This document is my project  project write-up. It contains answers to different questions links and commands used to transform models to IR.
 


## Explaining Custom Layers

The process behind converting custom layers involves  :
- Register the custom layer for the Model Optimizer tool. This process which is required to generate correct IR file with customs layers.
- As the Inference Engine only support CPU and GPU, the second step is to implement the kernel depending of target device.
For GPU acceleration support  we use OpenCL, C++ for CPU device.
- Register the kernel in Inference Engine and call to the custom kernel each time the IE meets the layer of the specific type in IR.

Some of the potential reasons for handling custom layers are the difference in topologies.
Custom layers  are used to quickly implement missing layers for cutting-edge topologies.
I compared models before and after conversion to Intermediate Representations by using Google Object Detection Api. 
The size of the model pre- and post-conversion was 112 KB
The inference time of the model pre- and post-conversion was 45ms, 70ms.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :
- Social distancing : In case to prevent the spread of a contagious disease, or to improve productivity in industries.
- In Supermarket : improve the number of person in front of cashier 
- Video surveillance : control restricted zones where persons cannot have access and it is not possible to use cloud based videos.
- Medical assistance : person having mobility problem can be assisted with such application by alerting when they are moving in some area.  

Each of these use cases would be useful because the application can use alert when in some conditions. 
Theses use cases can be implement at low cost in edge.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows  :
Poor quality lighting can reduce the accuracy and poor quality result because 
for computer vision application the image is essential, and the mean Average precision (mAp), can be reduced if some image are not recognised during inference.
The image size/focal length also have effect because the models have inputs requirement(inputs size of images).

## Model used in the application

I used the “Faster RCN Inception V2 COCO” model available with Tensorflow Object Detection API.

Here are list of commands used on model to convert to IR.

`$ cd /opt/intel/openvino/deployment_tools/model_optimizer`
`$ python3 mo_tf.py  --input_model /home/workspace/resources/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb  --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/resources/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels  -o /home/workspace/resources/`


Running example with Pedastrian_Detect_2_1_1.mp4
`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/resources/frozen_inference_graph.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`