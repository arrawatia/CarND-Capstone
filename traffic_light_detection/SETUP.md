
# Setup

The setup instructions here helped a lot in getting Tensorflow to work - [Alex Lechner's setup guide]

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



# Training Setup

- Get data
    ```
    mkdir data-downloads
    wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip\?dl\=0
    unzip alex-lechner-udacity-traffic-light-dataset.zip?dl=0 -O data-downloads/alex-lechner-udacity-traffic-light-dataset.zip
    ```
    
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


# Extract data from Site Testing Rosbag 
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

