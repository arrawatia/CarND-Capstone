
# Setup

[Alex Lechner's setup guide]

- Install python dependencies
    ```
    pip install -r requirements.txt
    ```

- Install protobuf compiler  
    ```
    brew install protobuf
    ```
    
    ```
    sudo apt-get install protobuf-compiler \
        python-pil \
        python-lxml \
        python-tk
    ```

- Get the tensorflow models. 

    ```
    git clone https://github.com/tensorflow/models.git tf-models && \
        cd tf-models && \
        git checkout f7e99c0 && \
        cd -
    ```


- Generate python code from protobufs
    ```
    cd tf-models/research && \
    protoc object_detection/protos/*.proto --python_out=. &&\
    cd -
    ```
- Add the generated code to the `PYTHONPATH`
    ```
    export PYTHONPATH=$PYTHONPATH:$PWD/tf-models/research:$PWD/tf-models/research/slim
    ```

- Test everything is setup correctly
    ```
    python tf-models/research/object_detection/builders/model_builder_test.py
    ```

    On macos + virtualenv, run this first `echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc`


[//]: # (References)
[Alex Lechner's setup guide]:  https://github.com/alex-lechner/Traffic-Light-Classification#set-up-tensorflow


# Dataset

There are 2 sources of traffic light images, one for each scenario.

1. Site testing data: Udacity provided a ROSbag file from Carla
2. Udacity Unity Simulator: Save Traffic lights image data from Udacity's simulator in the code

I started by finishing up Udacity's Object detection lab and 
worked through the [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

### Step 1 : Get raw data

#### Simulator 
The code here gets the raw data for simulator. Link to the tl_detector code

#### Site testing rosbag
Run the following commands in separate terminals to extract the image data from the rosbag

- Run a roscore instance.
    ```
    roscore
    ```

- Replay the bag in a loop.
    ```
    rosbag play -l path/to/your_rosbag_file.bag
    ```

- Save the data
    ```
    rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_color
    ```

### Step 2: Label the data by hand 

Using the **LabelImg** tool as described in [Step by Step TensorFlow Object Detection API Tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e). 

### Step 3: Create a TFRecord file from the labelled dataset 
The TensorFlow Object Detection API requires all the labeled training data to be in `TFRecord` file format.  


I started doing this and then read a chat mentioning labelled datasets were made graciouly available by former students. 

I did not see any value in spending a significant amount of time on handlabelling the data.

I downloaded the following datasets (one for validation and one for training). The images in the dataset are labeled and both the datasets come with a `TFRecord` file.

Validation dataset
- [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)
    ```
    mkdir data-downloads
    wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip\?dl\=0
    unzip alex-lechner-udacity-traffic-light-dataset.zip?dl=0 -O data-downloads/alex-lechner-udacity-traffic-light-dataset.zip
    ```
Training dataset
- [coldknight's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)




# Training

- SSD models from the lab 
- COCO models
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



# Training Setup


- Create dirs 

    ```
    mkdir traffic-light-detection && cd traffic-light-detection
    mkdir -p models/train
    mkdir data
    mkdir config
    ```
    
- Move the labelmap and tfrecord files to `data`

    ```
    mv data-downloads/alex-lechner-udacity-traffic-light-dataset/udacity_label_map.pbtxt data/
    mv data-downloads/alex-lechner-udacity-traffic-light-dataset/*.record data/
    mv data-downloads/dataset-sdcnd-capstone/*.record data/
    ```

- Download the pretrained models

    ```
    cd models
    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz

    tar -xvf *.tar.gz
    ```

- Copy the file `train.py` from the `tf-models/research/object_detection` 

    ```
    cp tf-models/research/object_detection/train.py .
    ```

- Get the configs for these models from `tf-models/research/object_detection/samples/configs` and copy them to `config` folder. 

- Copy `export_inference_graph.py` from the `tf-models/research/object_detection` folder 

    ```
    cp tf-models/research/object_detection/export_inference_graph.py .
    ```

- Test by training the **Fast Resnet 101** model 
    - Change the following settings
        1. Change `num_classes: 90` to `num_classes: 4`
        2. Change `max_detections_per_class: 100` and `max_total_detections: 300` to `max_detections_per_class: 10` and `max_total_detections: 10`
        4. Change `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"` to `fine_tune_checkpoint: "models/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"`
        5. Change `num_steps: 200000` down to `num_steps: 200`
        6. Change  `PATH_TO_BE_CONFIGURED` placeholders in `input_path` and `label_map_path` to `data/real_data.record` and `data/udacity_label_map.pbtxt`

    - Train 

        ```
        python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/faster_rcnn_resnet101_coco.config
        ```

    - Freeze the graph
        ```
        python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-200 --output_directory models
        ```

# Jupyter lab

- Install Jupyterlab
    
    ```
    pip install jupyterlab
    ```
    
- Run

    ```
    export PYTHONPATH=$PYTHONPATH:$PWD/tf-models/research:$PWD/tf-models/research/slim:$PWD/tf-models/research/object_detection
    jupyter lab
    ```

