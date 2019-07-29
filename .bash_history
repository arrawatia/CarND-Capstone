cd 
ls -ltr
ls -ltrha
exit
cat ~/.bashrc 
cat ~/.bash_rc 
cp ~/.bashrc .
ls
ls -a
cd ..
cp ~/.bashrc .
exit
roscore
rosmsg info geometry_msgs/Twist
rostopic list
exit
apt-get install ros-kinetic-turtlesim
env
rosrun turtlesim turtlesim_node
exti
exit
roscore
rosrun
roscore
exit
rosrun turtlesim turtle_teleop_key
/opt/ros/kinetic/lib/gazebo_ros/gzserver -e ode /capstone/catkin_ws/src/simple_arm/worlds/willow_garage.world __name:=gazebo __log:=/root/.ros/log/294a6040-ae9d-11e9-a77e-0242ac110002/gazebo-4.log
/opt/ros/kinetic/lib/gazebo_ros/gzserver
glxinfo | less
glxinfo | more
glxinfo | grep -i software
apt-get update && apt-get install -y gazebo7 libignition-math2-dev
mkdir -p ~/ros_ws/src
/bin/bash -c "source /opt/ros/kinetic/setup.bash && \
                  cd ~/ros_ws/ && \
                  catkin_make && \
                  echo 'export GAZEBO_MODEL_PATH=~/ros_ws/src/kinematics_project/kuka_arm/models' >> ~/.bashrc && \
                  echo 'source ~/ros_ws/devel/setup.bash' >> ~/.bashrc"
cd ~/ros_ws/src && git clone https://github.com/udacity/test_repo_robond_robotic_arm_pick_and_place.git && d_robotic_arm_pick_and_place/kinematics_project/ . && d_robotic_arm_pick_and_place/
cd -
ls
cd ..
l
cd catkin_ws/
rosdep update && rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
rm -rf build devel &&                   catkin_make
roslaunch simple_arm robot_spawn.launch
source devel/setup.bash 
roslaunch simple_arm robot_spawn.launch
tail -f /root/.ros/log/328ece62-aea1-11e9-97a6-0242ac110002/gazebo-5*
tail -f /root/.ros/log/328ece62-aea1-11e9-97a6-0242ac110002/
apt-get update &&   apt-get -y install libgl1-mesa-glx libgl1-mesa-dri 
glxgears
LIBGL_DEBUG=verbose glxgears
apt-get intall -y mesa-utils and libgl1-mesa-glx
apt-get install -y mesa-utils and libgl1-mesa-glx
LIBGL_DEBUG=verbose glxgears
exit
LIBGL_DEBUG=verbose glxgears
apt-get install locate
LIBGL_DEBUG=verbose glxgears
locate swrast_dri.so
apt-get install --reinstall libgl1-mesa-dri
locate swrast_dri.so
LIBGL_DEBUG=verbose glxgears
ls /usr/lib/x86_64-linux-gnu/dri/
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa
/usr/lib/x86_64-linux-gnu/mesa
ls /usr/lib/x86_64-linux-gnu/mesa
LIBGL_DEBUG=verbose glxgears
gazebo
find /usr -iname "*libGL.so*" -exec ls -l -- {} + 
LIBGL_DEBUG=verbose glxgears
glxinfo | grep direct
lspci | grep VGA
export LIBGL_DRIVERS_PATH=/usr/lib32/dri
ls /usr/lib32/dri
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/
glxinfo | grep direct
LIBGL_DEBUG=verbose glxinfo | grep direct
nvidia-smi
ldconfig -p | grep -i gl.so
exit
rostopic list
source devel/setup.bash 
rostopic echo /final_waypoints 
rosnode list
less src/waypoint_updater/waypoint_updater.py 
cat src/waypoint_updater/waypoint_updater.py a
exit
ls
source devel/setup.bash
git
git 
git branches
git branch
catkin_make
roslaunch launch/styx.launch
exit
ls -ltr
source devel/setup.bash
roslaunch launch/styx.launch
env 
source /opt/ros/kinetic/setup.bash 
source devel/setup.bash
env | grep -i ros
roslaunch launch/styx.launch
ls /opt/ros/kinetic/share
mv /opt/ros/kinetic/share/dbw_mkz_msgs src/
catkin_make
mv src/dbw_mkz_msgs /opt/ros/kinetic/share/
ls /opt/ros/kinetic/share/
ls /opt/ros/kinetic/share/ | grep dbw
ls -ltr
catkin_make
roslaunch launch/styx.launch
exit
