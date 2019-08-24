/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <msckf_vio/pose_graph_nodelet.h>

namespace msckf_vio {
void PoseGraphNodelet::onInit() {
  pose_graph_ptr.reset(new PoseGraph(getPrivateNodeHandle()));
  if (!pose_graph_ptr->initialize()) {
    ROS_ERROR("Cannot initialize PoseGraph ...");
    return;
  }
  return;
}

PLUGINLIB_DECLARE_CLASS(msckf_vio, PoseGraphNodelet,
    msckf_vio::PoseGraphNodelet, nodelet::Nodelet);

} // end namespace msckf_vio

