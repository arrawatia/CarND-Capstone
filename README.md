# Capstone project:  Programming a Real Self-Driving Car

The goal of this project is to program Udacity's self driving car to drive around a test lot. 

This requires completing the ROS subsystems for planning, perception and control. The perception subsystem classifies the traffic lights based on camera images. The navigation subsystem plans the path of the car and and updates the waypoints to make the car stop at red lights. The control subsytem uses "drive by wire" to move the car along waypoints at the correct velocity. 

I built the project in the 2 stages

1. Planning and Control
I completed the ROS modules for planning and control. The car drove successfully around the track while stopping at red lights. The perception module is mocked out using the data from the simulator.
2. Perception
I built a deep learning model for classifying traffic light images and used the model in the ROS module for traffic light detection. The car stopped at red lights by correcting classifying the images.


The document starts by describing the overall architecture of Carla. It discusses how all the subsystems interact and work together. 

It then describes the implementation of the planning and control modules followed by the implementation details of the traffic light classifier.


## Carla System Architecture

At a very high level, the architecture has 3 subsystems

1. Perception
2. Planning
3. Control

The subsystems react to data on the incoming ROS topics and publish their outputs on ROS topics for other modules to consume. The following diagram shows how all individual ROS nodes interact with each other.

![](imgs/final-project-ros-graph-v2.png)


## Implementation

### Planning

The goal is to finish the **Waypoint Updator** node so that it publishes a fixed number of waypoints ahead of the vehicle. It needs to decide on the correct velocities taking into account traffic lights and obstacles

This node will subscribe to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publish a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic.

Based on the walkthrough, this goal was achieved in 2 stages -

**Stage 1: Publish the waypoints**
The idea is to subscribe to `/base_waypoints` and `/current_pose` topic and publish a list of final waypoint data to `/final_waypoints`. The list of final waypoints is a subset of all waypoints published on `/base_waypoints` chosen to be in front of vehicle.

Upon completion of this step and the control module, the car is able to drive around the track in the simulator while ignoring traffice lights.


**Stage 2: Control the car velocity based on traffic lights**
The idea is to use information from `/traffic_waypoint` to change the waypoint target velocities. The car stops at red traffic lights and moves when light turns green.

Once traffic light detection is working properly, we can decelerate the car smoothly when an intersection with red traffic light is approaching.

### Control

The goal to implement a drive-by-wire (DBW) node which uses controllers to provide appropriate throttle, brake, and steering commands. After completing this step, the car is able to drive around the track in the simulator.

A safety driver may take control of the car during testing. If this happens, the DBW status will change and the change is published on `/vehicle/dbw_enabled` topic.

This requires changes to the DBW node which is responsible for steering the car. We change `dbw_node.py` to subscribe to the relevant topics and use the PID controller and lowpass filter in the twist controller (`twist_controller.py`) to calculate throttle, brake, and steering commands. We publish these commands to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics respectively.

### Perception

The goal of this module is to 

2 modules:

1. Obstacle detection 

 tl_detector.py. This node takes in data from the /image_color, /current_pose, and /base_waypoints topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic.

Detection: Detect the traffic light and its color from the /image_color. The topic /vehicle/traffic_lights contains the exact location and status of all traffic lights in simulator, so you can test your output

Waypoint publishing: Once you have correctly identified the traffic light and determined its position, you can convert it to a waypoint index and publish it.


2. Traffic light detection


- Uses deep learning. 
- 

I started by finishing up Udacity's Object detection lab and 
worked through the [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

Classification

The classification output has four categories: Red, Green, Yellow and off. To simplify, the final output will be Red or Non-Red, that is only the Red will be classified as TrafficLight.RED, and the other cases will be classified as TrafficLight.GREEN.

- Build the dataset 
    1. Collect images There are 2 sources of traffic light images, one for each scenario.

        1. Site testing data: Udacity provided a ROSbag file from Carla
        The code here gets the raw data for simulator. Link to the tl_detector code

        2. Udacity Unity Simulator: Save Traffic lights image data from Udacity's simulator in the code
        Extract it from the site testing ROS bag. Link to setup




-  Step 2: Label the data by hand 
    Using the **LabelImg** tool as described in [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e). 

- Step 3: Create a TFRecord file from the labelled dataset 
The TensorFlow Object Detection API requires all the labeled training data to be in `TFRecord` file format.  


I started doing this and then read a chat mentioning labelled datasets were made graciouly available by former students. 

I did not see any value in spending a significant amount of time on handlabelling the data.

I downloaded the following datasets (one for validation and one for training). The images in the dataset are labeled and both the datasets come with a `TFRecord` file.

    - Validation dataset [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)
    - Training dataset [coldknight's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)




#### Training

- SSD models from the lab 
    The task of object detection is to identify "what" objects are inside of an image and "where" they are. Given an input image, the algorithm outputs a list of objects, each associated with a class label and location (usually in the form of bounding box coordinates). In practice, only limited types of objects of interests are considered and the rest of the image should be recognized as object-less background.

Single-Shot Detector

Let's first remind ourselves about the two main tasks in object detection: identify what objects in the image (classification) and where they are (localization). In essence, SSD is a multi-scale sliding window detector that leverages deep CNNs for both these tasks.

A sliding window detection, as its name suggests, slides a local window across the image and identifies at each location whether the window contains any object of interests or not. Multi-scale increases the robustness of the detection by considering windows of different sizes. Such a brute force strategy can be unreliable and expensive: successful detection requests the right information being sampled from the image, which usually means a fine-grained resolution to slide the window and testing a large cardinality of local windows at each location.


- COCO models
detection models pre-trained on the COCO dataset,
They are also useful for initializing your models when training on novel datasets.

- Transfer learning - change the output from 

    - Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.
    It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

    - modify the pre-trained model that was designed to work on the 90 classes of the COCO dataset, to work on the 4 classes of my new dataset? y remove the last 90 neuron classification layer of the network and replace it with a new layer. To accomplish this with the object detection API, all you need to do is modify one line in the models config file. 
    

Started with 


Faster_RCNN_Inception_ResNet 
(http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

RFCN_ResNet101 
[faster rcnn resnet101]: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

Then moved to [ssd inception 171117]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
                                ms
ssd_inception_v2_coco	        42	24	Boxes
faster_rcnn_inception_v2_coco	58	28	Boxes
faster_rcnn_resnet101_coco	    106	32	Boxes
ssdlite_mobilenet_v2_coco	27	22	Boxes

SSD Inception V2 Coco (17/11/2017) Pro: Very fast, Con: Not good generalization on different data
SSD Inception V2 Coco (11/06/2017) Pro: Very fast, Con: Not good generalization on different data
Faster RCNN Inception V2 Coco (28/01/2018) Pro: Good precision and generalization of different data, Con: Slow
Faster RCNN Resnet101 Coco (11/06/2017) Pro: Highly Accurate, Con: Very slow


Take a look at the 

# Reflection
Bittersweet 
- Rewarding and satisfying at times and very frustrating other times
- Learning about ROS was fun. It is industrial grade 
- appreciated the architecture 
- very tricky to get it to work (No mac support, VM flaky, DBW node missing) 

- Transfer learning
- very practical
- But training models / validating / testing is very tedious
- Hard to get the correct dependencies of TF + models

- 3 term structure was better. This is 1-term's worth of work. Felt rushed. Walkthroughs helped but would have liked to figure it out on my own.