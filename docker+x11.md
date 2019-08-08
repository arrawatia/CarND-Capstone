How to run x11 with docker ?


1. Install Xquartz `brew install xquartz`
2. Run `/usr/X11/bin/xhost +`
3. 

```bash
IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
```

```bash
docker run -d --name firefox -e DISPLAY=$IP:0 -v /tmp/.X11-unix:/tmp/.X11-unix jess/firefox
```

```bash
docker run \
    --name capstone \
    -d \
    -p 4567:4567 \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/ \
    -v $PWD:/capstone \
    -v $PWD/log:/root/.ros/ \
    -v $PWD/.bash_history:/root/.bash_history \
    -v $PWD/.bashrc:/root/.bashrc \
    capstone -- tail -f /dev/null
```

```bash
docker exec -it capstone bash
source /opt/ros/kinetic/setup.bash
```