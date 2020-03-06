#!/usr/bin/env python
#author: qinlin@andrew.cmu.edu


import rosbag
from transformations import *
from scipy import signal
from datetime import datetime
bag = rosbag.Bag('rounded_square.bag')
print bag.get_type_and_topic_info()

bag_start = bag.get_start_time()
bag_end = bag.get_end_time()

cmd_vel, cmd_steering_angle, cmd_t = list(), list(), list() 
for topic, msg, t in bag.read_messages(topics=['/commands/keyboard']):
    #cmd_vel.append(msg.drive.speed)
    #cmd_steering_angle.append(msg.drive.steering_angle)
    cmd_t.append(t)
    #print msg


pos_x, pos_y, yaw, pos_t = list(), list(), list(), list() 


for topic, msg, t in bag.read_messages(topics=['/ekf_localization/odom']):
    #pos_x.append(msg.pose.pose.position.x)
    #pos_y.append(msg.pose.pose.position.y)
    pos_t.append(t)

waypoints_x, waypoints_y, waypoints_curvature, waypoints_t = list(), list(), list(), list()
for topic, msg, t in bag.read_messages(topics=['/aa_planner/waypoints']):#/aa_planner/waypoints
    waypoints_t.append(t)
    #print msg

cut_start_time = max(cmd_t[0], pos_t[0], waypoints_t[0])
cut_end_time = min(cmd_t[-1], pos_t[-1], waypoints_t[-1])


for topic, msg, t in bag.read_messages(topics=['/ekf_localization/odom'], start_time = cut_start_time, end_time = cut_end_time):
    pos_x.append(msg.pose.pose.position.x)
    pos_y.append(msg.pose.pose.position.y)
    pos_z = msg.pose.pose.orientation.z
    pos_w = msg.pose.pose.orientation.w
    yaw.append(euler_from_quaternion([pos_x[-1], pos_y[-1], pos_z, pos_w])[2])
    pos_t.append(t)

for topic, msg, t in bag.read_messages(topics=['/aa_planner/waypoints'], start_time = cut_start_time, end_time = cut_end_time):
    waypoints_x.append(msg.x)
    #print msg.x
    waypoints_y.append(msg.y)
    waypoints_curvature.append(msg.z)

for topic, msg, t in bag.read_messages(topics=['/commands/keyboard'], start_time = cut_start_time, end_time = cut_end_time):
    cmd_vel.append(msg.drive.speed)
    cmd_steering_angle.append(msg.drive.steering_angle)

print len(pos_x), len(pos_y), len(yaw), len(waypoints_x), len(waypoints_y), len(waypoints_curvature), len(cmd_vel), len(cmd_steering_angle)
pos_x_resample = signal.resample(pos_x, len(cmd_vel))
pos_y_resample = signal.resample(pos_y, len(cmd_vel))
yaw_resample = signal.resample(yaw, len(cmd_vel))
waypoints_curvature_resample = signal.resample(waypoints_curvature, len(cmd_vel))
duration_sec = (cut_end_time - cut_start_time).to_sec()
frequency = len(cmd_vel)/duration_sec
print frequency
#print type(pos_x_resample)
#print (cut_end_time - cut_start_time).to_sec()
#print (cut_end_time - cut_start_time)
