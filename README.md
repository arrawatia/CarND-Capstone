# Capstone project:  Programming a Real Self-Driving Car

Individual submission by Sumit Arrawatia (sumit.arrawatia@gmail.com)

The goal of this project is to program Udacity's Self-Driving Car (Carla) to drive around a test lot. 

This requires completing the ROS subsystems for planning, perception and control. The perception subsystem classifies the traffic lights based on camera images. The planning subsystem plans the path of the car and and updates the waypoints to make the car stop at red lights. The control subsytem uses "drive by wire" to move the car along waypoints at the correct velocity. 

I built the project in the 2 stages

**1. Planning and Control**

I completed the ROS modules for planning and control. The car drove successfully around the track while stopping at red lights. The perception module is mocked out using the data from the simulator.

**2. Perception**

I built a deep learning model for classifying traffic light images and used the model in the ROS module for traffic light detection. The car stopped at red lights by correcting classifying the images.


The document starts by describing the overall architecture of Carla. It discusses how all the subsystems interact and work together. It then describes the implementation of the planning and control modules followed by the implementation details of the traffic light classifier.


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

This requires changes to the DBW node which is responsible for steering the car. We change `dbw_node.py` to subscribe to the relevant topics and use the PID controller and lowpass filter in the twist controller (`twist_controller.py`) to calculate throttle, brake, and steering commands. These commands are published to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics respectively. 

### Perception

The goal of this module is to make sure that the car stops for red traffic lights. This is done in two parts:

**1. Find the upcoming traffic light** 

We use the vehicle location and find the closest visible traffic light ahead in the `process_traffic_lights` method of `tl_detector.py`. We find the closest waypoints to the vehicle and lights and then use this to find which light is ahead of the vehicle along the list of waypoints.

**2. Classify the color of the light using the camera images**

I used a deep learning classifier to classify the entire image into one of 4 categories: a red light, yellow light, green light, or no light. 

This is described in the Deep Learning model section below.


### Deep Learning model for Traffic Light Detection

The goal is to classify the image into one of the 4 categories

1. Red Light
2. Green Light
3. Yellow Light
4. No Light

I started by finishing up Udacity's Object detection lab and 
worked through the [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

The model was built in the following steps:

#### Step 1: Build the dataset

We need data for training the deep learning model. We have 2 sources of data: images from the simlulator and actual site images in the form of a rosbag.

After we get the data, it has to be labelled manually. The  [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e) describes a way to do this using the **LabelImg** tool.
 
Once the data is collected and labelled, we need to convert the data into `TFRecord` format. The TensorFlow Object Detection API requires all the labeled training data to be in `TFRecord` file format.  

I started doing this but found that labelled datasets were made graciouly available by former students. So, I decided to use two out of many available datasets - one each for validation and training. 
 
- Training - [coldknight's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)

- Validation - [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)


#### Step 2 : Choose a model 

Our model needs to detect traffic lights in the image and classify the light color, if it finds one.

So, we need a model that can detect objects or identify "what" objects are inside of an image (classification) and "where" they are (localization). Given an input image, the model should give us a list of objects, their classes and bounding box coordinates. 

I used a **Single-Shot Detector (SSD)** model to do this. A SSD is a multi-scale sliding window detector that uses deep CNNs for both classification and localization. It slides a local window across the image and identifies at each location whether the window contains any object of interests or not. Multi-scale increases the robustness of the detection by considering windows of different sizes. 

Rather than training and building such a model from scratch, I used **Transfer Learning** to adapt one of pre-trained **Object Detection** models from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). I chose to modify one of the pre-trained model that was designed to work on the 90 classes of the COCO dataset. I modified it to work on the 4 classes of the traffic light datasets.

The Transfer Learning process is described in the **Training Setup** section of the [setup guide](setup.md).

I started by using the [Faster RCNN Resnet101 Coco ](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) and [Faster RCNN Inception V2 Coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) models used in the Object Detection lab. These models have really good accuracy but I could not get these models to perform fast enough to be usable in simulator testing. 


I read that others were having success using [SSD Inception V2 Coco]( http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) and moved to this as well. This model is sufficiently fast in testing but doesnot generalize well. So, I trained a different model for each (simulator and site) dataset.

#### Step 3: Train / Validate / Save the model

The model was trained on AWS using the [coldknight's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset) and validated on [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)

The final model was then frozen and used in simulator testing.

This jupyter notebook [traffic_light_detection/traffic_lights.ipynb](traffic_light_detection/traffic_lights.ipynb) shows the models being used to classify 10 random images from both simulator and site datasets.

## Reflection
This project was a bittersweet experience. It was rewarding and satisfying at times and very frustrating other times.

Learning about ROS was fun. It is interesting to work with a industrial grade framework and I appreciated the modular architecture. Once I got a working setup with Docker, the developing and debugging cycle went much faster.

But it was very tricky to get it to work. I was working on a Macbook and the Udacity VM was very flaky. I decided to investigate the Docker setup but had to solve issues with the install, port forwarding, the ROS cmake compilation and getting the simulator to work. 

Using Transfer learning was a practical way to get a usable model in resonable amount of time. It was interesting to try out various models to see which one worked best. I had never needed to think about speed of inference in the first term and it was a good lesson on the practical constraints of using these models for real-world scenarios.

But the entire cycle of training + validating + testing models is very tedious. This was my least favorite part of Term 1 projects too. It was very hard to get the correct dependencies of TF 1.3 to match up the prebuilt models.

I feel the 3 term structure was better. This project requires 1-term's worth of work. I felt really rushed doing this project. The walkthroughs helped but I would have liked to figure it out on my own.