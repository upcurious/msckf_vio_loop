/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef POSE_GRAPH_NODELET_H
#define POSE_GRAPH_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <msckf_vio/pose_graph.h>

namespace msckf_vio {
class PoseGraphNodelet : public nodelet::Nodelet {
public:
  PoseGraphNodelet() { return; }
  ~PoseGraphNodelet() { return; }

private:
  virtual void onInit();
  PoseGraphPtr pose_graph_ptr;
};
} // end namespace msckf_vio

#endif

