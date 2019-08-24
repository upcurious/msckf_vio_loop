# msckf_vio_loop
This is msckf_vio with loop detection and global pose graph optimization

How to run:
1、catkin_make --pkg msckf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
2、roslaunch msckf_vio msckf_vio_euroc.launch
3、roslaunch msckf_vio msckf_rviz.launch
4、rosbag play /home/yourname/Downloads/MH_05_difficult.bag
