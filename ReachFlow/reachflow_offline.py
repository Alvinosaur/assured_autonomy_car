#!/usr/bin/env python
#author: qinlin@andrew.cmu.edu
import os
import subprocess
from PIL import Image
from mpmath import *
import copy
from interval import interval, inf, imath#
from transformations import *
import rospy
import ackermann_msgs.msg
import geometry_msgs.msg
import nav_msgs.msg
import numpy as np
#import reach_flow.msg
#import FlowstarVisual.msg
import time
import timeit
import datetime
from pyqtgraph.pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.pyqtgraph as pg
from uncertainty_estimation_gp import *
L = 0.257
L_f = 0.137
L_r = 0.12
C_alpha = 56.4
I_z = 0.0558
mode = 'rounded_square'#'circle', 'straight',rounded_square

# Change below according to your config
reach_flow_dir = '/home/alvin/Projects/assured_autonomy_car/ReachFlow'

# save safety invalidations
SAVE_FILES = False


# need to change the os path to reach_flow/src for the file to be called through roslaunch
#-amankh
try:
        os.chdir(reach_flow_dir)
        print("Working directory changed to reah_flow/src")
except OSError:
	print("Couldn't change the working directory to reach_flow/src. This code will not work with roslaunch or rosrun file")

class Nodo(object):
    def __init__(self, graph_widget, ifplot=True):
        global mode
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.x_dot = 0
        self.y_dot = 0
        self.yaw_dot = 0
        self.command_speed = 0
        self.command_angle = 0
        self.loop_rate = rospy.Rate(10)
        self.yaw = 0
        self.waypoint_curvature = 0
        self.gp_data = np.load('K_inv_3d.npz')
        self.K_vel_inv = self.gp_data['K_vel_inv']
        self.K_angle_inv = self.gp_data['K_angle_inv']
        self.X_train = self.gp_data['X_train']
        self.Y_train_vel = self.gp_data['Y_train_vel']
        self.length_vel = self.gp_data['length_vel']
        self.Y_train_angle = self.gp_data['Y_train_angle']
        self.length_angle = self.gp_data['length_angle'] 

        self.ifplot = ifplot
        self.box_markers = []
        self.state_marker = None
        if self.ifplot: self.init_plot(graph_widget)
        
        if mode == 'rounded_square':
            self.waypoint_x = 1
            self.waypoint_y = 0
        else:
            self.waypoint_x = 0
            self.waypoint_y = 0
        self.state = [self.pos_x, self.pos_y, self.yaw, self.x_dot, self.y_dot, self.yaw_dot]
        self.command = [self.command_speed, self.command_angle]
        rospy.Subscriber("ekf_localization/odom", nav_msgs.msg.Odometry, self.stateCallback)
        rospy.Subscriber("commands/keyboard", ackermann_msgs.msg.AckermannDriveStamped, self.actionCallback)
        rospy.Subscriber("aa_planner/waypoints", geometry_msgs.msg.Point, self.waypointCallback)
        self.start()
        #rospy.Subscriber("aa_planner/commands", ackermann_msgs.msg.AckermannDriveStamped, self.actionCallback)
	#self.flow_pub = rospy.Publisher("reach_flow/flow", reach_flow.msg.FlowstarVisual, queue_size=1)
        #self.action_pub = rospy.Publisher("commands/keyboard", ackermann_msgs.msg.AckermannDriveStamped, queue_size=1)

    def stateCallback(self,state_t):
        self.pos_x = state_t.pose.pose.position.x#same to odom.pose.pose.position.x?
        self.pos_y = state_t.pose.pose.position.y
        self.yaw = euler_from_quaternion([self.pos_x,self.pos_y,state_t.pose.pose.orientation.z,state_t.pose.pose.orientation.w])[2]
        self.x_dot = state_t.twist.twist.linear.x
        self.y_dot = state_t.twist.twist.linear.y
        self.yaw_dot = state_t.twist.twist.linear.z
    def actionCallback(self,action_t):
        self.command_speed = action_t.drive.speed
        self.command_angle = action_t.drive.steering_angle

    def waypointCallback(self,waypoint_t):
        self.waypoint_curvature = waypoint_t.z
        self.waypoint_x = waypoint_t.x
        self.waypoint_y = waypoint_t.y
    def init_plot(self, graph_widget):
        self.p = graph_widget.addPlot(row=0, col=0)

        if mode == "circle":
            th = np.arange(0*np.pi, 2*np.pi+np.pi/10, np.pi/10)
            self.p.plot(x=1 * np.cos(th), y=1 * np.sin(th)+1)
            self.p.setXRange(-2, 2, padding=0)
            self.p.setYRange(1, 3, padding=0)
        if mode == "rounded_square":
            self.p.plot(x=np.arange(0, 1.1, 0.1), y=0*np.arange(0, 1.1, 0.1))#1st

            th = np.arange(1.5*np.pi, 2*np.pi+np.pi/10, np.pi/10)
            self.p.plot(x=1 * np.cos(th)+1, y=1 * np.sin(th)+1)#2nd

            self.p.plot(x=2+0*np.arange(1, 2.1, 0.1), y=np.arange(1, 2.1, 0.1))#3rd

            th = np.arange(0*np.pi, 0.5*np.pi+np.pi/10, np.pi/10)
            self.p.plot(x=1 * np.cos(th)+1, y=1 * np.sin(th)+2)#4th

            self.p.plot(x=np.arange(0, 1.1, 0.1), y=3+0*np.arange(0, 1.1, 0.1))#5th               

            th = np.arange(0.5*np.pi, 1.0*np.pi+np.pi/10, np.pi/10)
            self.p.plot(x=1 * np.cos(th)+0, y=1 * np.sin(th)+2)#6th

            self.p.plot(x=-1+0*np.arange(0, 1.1, 0.1), y=np.arange(1, 2.1, 0.1))#7th

            th = np.arange(np.pi, 1.5*np.pi+np.pi/10, np.pi/10)
            self.p.plot(x=1 * np.cos(th)+0, y=1 * np.sin(th)+1)#8th
            self.p.setXRange(-2, 5, padding=0)
            self.p.setYRange(-2, 5, padding=0)
            
    def start(self):
        total_t = 0
        counter = 0
        horizon = 5
        max_t = 0
        min_t = 100
        MU_V = 0  #parameters from sampling approach
        SIGMA_V = 0.04
        MU_DELTA = 0.002
        SIGMA_DELTA = 0.27
        c_sigma = 1
        V_SAMPLING_BOUND = MU_V+c_sigma*SIGMA_V
        DELTA_SAMPLING_BOUND = MU_DELTA+c_sigma*SIGMA_DELTA 
        counter_timeout = 0
        bad_position_x = []
        bad_position_y = []
        counter_inside = 0#how many future state is actually included by reachability 
        safety_violated = False# set true if unsafe
        uncertainty_estimation_method = 'gp' #option['gp', 'sampling']SUE-GP and sampling distribution approach
        x_list_buffer, y_list_buffer = [], []
        contract_ratio_speed = 0
        contract_ratio_delta = 0
        counter_contract_speed = 0
        counter_contract_delta = 0
        model = 'nonlinear_with_beta' # option ['nonlinear_with_beta', 'linear', 'nonlinear_with_beta', 'linear_tire']
        counter_eg_x, counter_eg_y = [], []
        counter_flow_x, counter_flow_y = [], []
        x_all, y_all, ind_all, flow_x_all, flow_y_all = [], [], [], [], []
        delta_t_all = []

        total_setup = 0
        total_flowstar = 0
        total_parse_flowstar = 0
        total_plot_time = 0

        bag_start = time.time()

        # while (bag_start - time.time()) < 20:  # short test
        while not rospy.is_shutdown():
            counter = counter+1

            start = time.time()
            if self.ifplot == True:
                if self.state_marker is not None:
                    self.p.removeItem(self.state_marker)
                for marker in self.box_markers:
                    self.p.removeItem(marker)
                self.box_markers = []
            end = time.time()
            total_plot_time += (end - start)

            self.state = [self.pos_x, self.pos_y, self.yaw, self.x_dot, self.y_dot, self.yaw_dot]
            self.command = [self.command_speed, self.command_angle]
	    #####################simulation of malicious command#################
            #if np.abs(self.pos_x-1)+np.abs(self.pos_y-0)<=0.1 and (self.waypoint_x,self.waypoint_y) == (2, 1):
            #    print '################reach position###############'
            #    print self.state, self.command, self.waypoint_x, self.waypoint_y
            #    #break
            #    #self.command = [2, 0]
            
            start = time.time()
            if uncertainty_estimation_method == 'sampling':
                speed_bound = V_SAMPLING_BOUND #time invariant uncertainty of speed change in future 1s
                delta_bound = DELTA_SAMPLING_BOUND #time invariant uncertainty of angle change in future 1s
            if uncertainty_estimation_method == 'gp':
                input_data = [self.pos_x-self.waypoint_x, self.pos_y-self.waypoint_y, self.waypoint_curvature]
                X_new = np.array(input_data).reshape(1, len(input_data))
                mu, cov = posterior_predictive_noise_free(self.K_vel_inv, X_new, self.X_train, self.Y_train_vel, self.length_vel)
                mu_vel = np.abs(mu[0][0])
                sigma_vel = np.abs(cov[0][0])
                speed_bound = min(mu_vel+1*sigma_vel, V_SAMPLING_BOUND)
                if speed_bound < V_SAMPLING_BOUND:
                    counter_contract_speed+=1 
                    contract_ratio_speed += np.abs(speed_bound-V_SAMPLING_BOUND)
                mu, cov = posterior_predictive_noise_free(self.K_angle_inv, X_new, self.X_train, self.Y_train_angle, self.length_angle)
                mu_angle = np.abs(mu[0][0])
                sigma_angle = np.abs(cov[0][0])
                delta_bound = min(mu_angle+1*sigma_angle, DELTA_SAMPLING_BOUND)
                if delta_bound < DELTA_SAMPLING_BOUND:
                    counter_contract_delta+=1 
                    contract_ratio_delta += np.abs(delta_bound-DELTA_SAMPLING_BOUND)
            end = time.time()
            setup_time = end - start
            total_setup += setup_time
 
            start = time.time()
            try:
                flow = executeFlowstar(model, horizon, self.state, self.command, self.waypoint_x, self.waypoint_y, speed_bound, delta_bound)
            except Exception as e:
                print("Error in executeFlowstar: %s" % e)
                continue
            end = time.time()
            flowstar_time = end - start
            total_flowstar += flowstar_time
             
            start = time.time()
            flow_x = flow[0]
            flow_y = flow[1]
            x_list = []
            y_list = []
            x_list_i, y_list_i = [], []
            
            for i in range(len(flow_x.split(';'))):
                x_temp = flow_x.split(';')[i]
                y_temp = flow_y.split(';')[i]
                x_list.append(x_temp.split(','))
                y_list.append(y_temp.split(','))
            
            for i in range(len(x_list)):
                if len(x_list[i]) != 5 and len(y_list[i]) != 5: continue
                x0, x1, x2, x3, x4 = x_list[i]
                y0, y1, y2, y3, y4 = y_list[i]
                box_line1 = self.p.plot(x=[float(x0), float(x1)], y=[float(y0), float(y1)])
                box_line2 = self.p.plot(x=[float(x1), float(x2)], y=[float(y1), float(y2)])
                box_line3 = self.p.plot(x=[float(x2), float(x3)], y=[float(y2), float(y3)])
                box_line4 = self.p.plot(x=[float(x3), float(x4)], y=[float(y3), float(y4)])
                self.box_markers += [box_line1, box_line2, box_line3, box_line4]
            end = time.time()
            total_plot_time += (end - start)

            start = time.time()
            x_list_buffer.append(x_list_i)
            y_list_buffer.append(y_list_i)
            if counter >= horizon:
                del x_list_buffer[0]
                del y_list_buffer[0] 
                predict_indicator = check_inside_flow(x_list_buffer, y_list_buffer, self.state)
                x_all.append(self.pos_x)
                y_all.append(self.pos_y)
                ind_all.append(predict_indicator)
                flow_x_all.append(x_list_buffer[0])
                flow_y_all.append(y_list_buffer[0]) 
                if predict_indicator == True:
                    counter_inside = counter_inside+1
                else:
                    bad_position_x.append(self.pos_x)
                    bad_position_y.append(self.pos_y)
                    counter_eg_x.append(self.pos_x)
                    counter_eg_y.append(self.pos_y)
                    counter_flow_x.append(x_list_buffer[0])
                    counter_flow_y.append(y_list_buffer[0])
                    if SAVE_FILES:
                        np.savez('counter_eg' +str(len(counter_eg_x)) + '.npz', counter_eg_x = counter_eg_x, counter_eg_y=counter_eg_y,counter_flow_x=counter_flow_x, counter_flow_y=counter_flow_y)
            
            end = time.time()
            parsing_flowstar_time = end - start
            total_parse_flowstar += parsing_flowstar_time

            # start = time.time()
            start = time.time()
            self.state_marker = pg.ScatterPlotItem(x=[self.state[0]], y=[self.state[1]], size=12)
            self.p.addItem(self.state_marker)
            pg.QtGui.QApplication.processEvents()
            end = time.time()
            total_plot_time += (end - start)

            # delta_t = (end-start).total_seconds()*1000
            # delta_t_all.append(delta_t)
            # total_t = total_t+delta_t  
            # if delta_t > 100:
            #     counter_timeout += 1
            #     print counter, 'time out!!!!'  
            #     print 'timeout percentage %: ', 100*counter_timeout/counter    
            # if counter > 0:
            #     ave_t = total_t/counter
            # if delta_t >= max_t:
            #     max_t = delta_t
            # if delta_t <= min_t:
            #     min_t = delta_t
            # if counter%1000 == 0:
            #     print('runtime', np.max(delta_t_all), np.min(delta_t_all), np.mean(delta_t_all), np.std(delta_t_all))
            #     print('inclusion', counter_inside, counter)
            #     print('contraction', contract_ratio_speed/counter_contract_speed, contract_ratio_delta/counter_contract_delta)
            #     #np.savez('all_data.npz', x_all=x_all, y_all=y_all, ind_all=ind_all, flow_x_all=flow_x_all, flow_y_all = flow_y_all)               
            #     break
            self.loop_rate.sleep()
        
        print("Total setup: %.3f" % (total_setup / counter))
        print("Total flowstar: %.3f" % (total_flowstar / counter))
        print("Total parse flowstar: %.3f" % (total_parse_flowstar / counter))
        print("Total plot: %.3f" % (total_plot_time / counter))
        print("Average runtime: %.3f" % ((time.time() - bag_start) / counter))



def check_inside_flow(xflow, yflow, state):
    '''
        Function: check if a state is enclosed in a reachable set
    '''
    x = state[0]
    y = state[1]
    x_history_flow = xflow[0]
    y_history_flow = yflow[0]
    indicator = False
    for i in range(len(x_history_flow)):
        x_min = x_history_flow[i][0]
        x_max = x_history_flow[i][1]
        y_min = y_history_flow[i][0]
        y_max = y_history_flow[i][2]          
        if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
            indicator = True
            break
        else:
            continue             
    return indicator
def executeFlowstar(model, horizon, state, command, waypoint_x, waypoint_y, speed_bound, delta_bound):
    '''
        Function: External execution of reachability computation by calling ./RC_bicycle
        Input: type of dynamic model, state, command, waypoint coordinate, control bounds
        Output: flowpipe points for visulization, safety indicator
    '''
    uncertainty_pos = 0.1
    horizon = 0.1*horizon
    uncertainty_speed_state = 0.001 #velocity state uncertainty
    uncertainty_angle_rate = 0.001 #yaw dot state uncertainty
    uncertainty_angle = (horizon/1)*delta_bound #control steering angle uncertainty, SUE-GP
    uncertainty_speed = (horizon/1)*speed_bound #control velocity uncertainty, SUE-GP
    epsilon_cir = 0.1
    epsilon_line = 0.5
    pos_x = [state[0]-uncertainty_pos, state[0]+uncertainty_pos]
    pos_y = [state[1]-uncertainty_pos, state[1]+uncertainty_pos]
    yaw = [state[2]-uncertainty_angle, state[2]+uncertainty_angle]
    x_dot = [positive(state[3]-uncertainty_speed_state), positive(state[3] +uncertainty_speed)]
    x_dot_I = interval([x_dot[0], x_dot[1]])
    y_dot = [positive(state[4]-uncertainty_speed), positive(state[4]+uncertainty_speed)]
    y_dot_I = interval([y_dot[0], y_dot[1]])
    yaw_dot = [positive(state[5]-uncertainty_angle_rate), positive(state[5]+uncertainty_angle_rate)]
    yaw_dot_I = interval([yaw_dot[0], yaw_dot[1]])
    command_speed = [command[0]-uncertainty_speed, command[0]+uncertainty_speed]
    delta_I = interval([command[1]-uncertainty_angle, command[1]+uncertainty_angle])
    
    alpha_f = initial_alpha_f_interval(y_dot_I, yaw_dot_I, x_dot_I, delta_I)
    alpha_r = initial_alpha_r_interval(y_dot_I, yaw_dot_I, x_dot_I)
 
    global mode
    if mode == 'rounded_square':
        if (waypoint_x,waypoint_y) == (1, 0):
            constant = epsilon_line 
            formula = str(constant)+"-y"
        if (waypoint_x,waypoint_y) == (2, 1):#x0=1,y0=1
            constant = -1-1+(epsilon_cir+1)*(epsilon_cir+1)
            formula = str(constant)+"+2*x+2*y-x*x-y*y"
        if (waypoint_x,waypoint_y) == (2, 2):
            constant = epsilon_line+2
            formula = str(constant)+"-x"
        if (waypoint_x,waypoint_y) == (1, 3):#x0=1,y0=2
            constant = (epsilon_cir+1)*(epsilon_cir+1)-5
            formula = str(constant)+"+2*x+4*y-x*x-y*y"
        if (waypoint_x,waypoint_y) == (0, 3):
            constant = epsilon_line+3
            formula = str(constant)+"-y"
        if (waypoint_x,waypoint_y) == (-1, 2):#x0=0,y0=2
            constant = (epsilon_cir+1)*(epsilon_cir+1)-4
            formula = str(constant)+"+4*y-x*x-y*y"
        if (waypoint_x,waypoint_y) == (-1, 1):
            constant = epsilon_line+1
            formula = str(constant)+"+x"
        if (waypoint_x,waypoint_y) == (0, 0) and state[2]!=0:#x0=0, y0=1
            constant = (epsilon_cir+1)*(epsilon_cir+1)-1
            formula = str(constant)+"+2*y-x*x-y*y"
        if (waypoint_x,waypoint_y) == (0, 0) and abs(state[2])<=0.1:
            constant = epsilon_line 
            formula = str(constant)+"-y"
        formula = "20-x"
    if mode == 'straight':
        constant = epsilon_line 
        formula = str(constant)+"-y"
        formula = "20-x"

    if mode == 'circle':
        constant = -1-epsilon_cir        
        #formula = str(constant)+"+x*x+y*y"
        formula = "20-x"

    exec_file = os.path.join(model, './RC_bicycle')
    ####################for model without beta#########################
    if model == 'nonlinear_without_beta':
        beta = initial_beta_interval(delta_I)
        args = [exec_file, str(pos_x[0]), str(pos_x[1]), str(pos_y[0]), str(pos_y[1]), 
            str(yaw[0]), str(yaw[1]), str(command_speed[0]), str(command_speed[1]),
            str(delta_I[0][0]), str(delta_I[0][1]), str(-1*speed_bound), str(speed_bound),
            str(-1*delta_bound), str(delta_bound), formula]

    ####################for model with beta############################
    
    elif model == 'nonlinear_with_beta':
        beta = initial_beta_interval(delta_I)
        args = [exec_file, str(pos_x[0]), str(pos_x[1]), str(pos_y[0]), str(pos_y[1]), 
            str(yaw[0]), str(yaw[1]), str(beta[0]), str(beta[1]), str(command_speed[0]), str(command_speed[1]),
            formula]

    elif model == 'linear':
        a13 = -1*command[0]*np.sin(state[2])
        a23 = command[0]*np.cos(state[2])
        b11 = np.cos(state[2])
        b21 = np.sin(state[2])
        b31 = np.tan(command[1])/0.12
        b32 = command[0]/(0.12*np.cos(command[1])*np.cos(command[1])) 
        args = [exec_file, str(a13), str(a23), str(b11), str(b21), str(b31), str(b32), 
            str(pos_x[0]), str(pos_x[1]), str(pos_y[0]), str(pos_y[1]), str(yaw[0]), str(yaw[1]),
            str(command_speed[0]), str(command_speed[1]), str(delta_I[0][0]), str(delta_I[0][1]),
            str(-1*speed_bound), str(speed_bound), str(-1*delta_bound), str(delta_bound)]

    else:
        beta = initial_beta_interval(delta_I)
        alpha_f = initial_alpha_f_interval(y_dot_I, yaw_dot_I, x_dot_I, delta_I)
        alpha_r = initial_alpha_r_interval(y_dot_I, yaw_dot_I, x_dot_I)
        Fxf = [0, 0]#fxi = C*kappa, 0 if no skid
        Fxr = [0, 0]
        Fyf = [min(-1*C_alpha*interval(alpha_f)[0][0], -1*C_alpha*interval(alpha_f)[0][1]), max(-1*C_alpha*interval(alpha_f)[0][0], -1*C_alpha*interval(alpha_f)[0][1])]
        Fyr = [min(-1*C_alpha*interval(alpha_r)[0][0], -1*C_alpha*interval(alpha_r)[0][1]), max(-1*C_alpha*interval(alpha_r)[0][0], -1*C_alpha*interval(alpha_r)[0][1])]
        args = [exec_file, str(pos_x[0]), str(pos_x[1]), str(pos_y[0]), str(pos_y[1]),
            str(yaw[0]), str(yaw[1]), str(yaw_dot[0]), str(yaw_dot[1]), str(Fxf[0]), str(Fxf[1]),
            str(Fxr[0]), str(Fxr[1]), str(Fyf[0]), str(Fyf[1]), str(Fyr[0]), str(Fyr[1]), 
            str(beta[0]), str(beta[1]), str(command_speed[0]), str(command_speed[1]), formula]
    
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    # output = ...0.918423,1.20878,1.20878,0.918423,0.918423,2.88629,2.88629,3.17285,3.17285,2.88629;0,0.4,0...
    (output, err) = process.communicate()
    flow_x = list()
    flow_y = list()
    color_list = list()
    output = output.decode(encoding='utf8')
    lines = output.split("\n")[:-1]
    for line in lines:
        points_str, color_str = line.split(";")
        points = points_str.split(",")
        colors = color_str.split(",")
        string_x = ",".join(points[0:5])
        string_y = ",".join(points[5:])

        if float(colors[0]) > 0:
            color_list.append('red')
        elif float(colors[2]) > 0:
            color_list.append('blue')
        else:
            color_list.append('green')
        flow_x.append(string_x)
        flow_y.append(string_y)
    if 'red' in color_list or 'blue' in color_list:
        indicator = 0#'unsafe'
    else:
        indicator = 1#'safe'
    flow_x_itr = ';'.join(flow_x)#x multiple flowsegments
    flow_y_itr = ';'.join(flow_y)#y multiple flowsegments
    return [flow_x_itr, flow_y_itr, indicator]
def positive(x):
    return max(x, 0)
def initial_beta_interval(delta_I):
    '''
       Function: compute the interval of beta for the kinematic bicycle model
       Input: steering angle interval
    '''
    global L
    global L_f
    global L_r
    #delta_0 = interval[delta-uncertainty, delta+uncertainty]
    beta = (lambda delta: imath.atan(L_r * imath.tan(delta)/L))(delta_I)
    beta_min = min(beta[0][0], beta[0][1])
    beta_max = max(beta[0][0], beta[0][1])
    return [beta_min, beta_max]
def initial_alpha_f_interval(v_y_I, yaw_dot_I, v_x_I, delta_I):
    '''
       Function: compute the interval of beta for the kinematic bicycle model
       Input: intervals of vy, yaw_dot, vx, delta
    '''
    global L
    global L_f
    global L_r
    fun = lambda v_y,yaw_dot,v_x,delta: imath.atan((v_y+L_f*yaw_dot)/v_x)-delta
    alpha_f = fun(v_y_I, yaw_dot_I, v_x_I, delta_I)
    return [alpha_f[0][0], alpha_f[0][1]]
def initial_alpha_r_interval(v_y_I, yaw_dot_I, v_x_I):
    '''
       Function: compute the interval of beta for the kinematic bicycle model
       Input: intervals of vy, yaw_dot, vx
    '''
    global L
    global L_f
    global L_r
    fun = lambda v_y,yaw_dot,v_x: imath.atan((v_y-L_f*yaw_dot)/v_x)
    alpha_r = fun(v_y_I, yaw_dot_I, v_x_I)
    return [alpha_r[0][0], alpha_r[0][1]]


if __name__=='__main__':
    app = QtGui.QApplication([])
    w = QtGui.QMainWindow()
    cw = pg.GraphicsLayoutWidget()
    w.show()
    w.resize(600,600)
    w.setCentralWidget(cw)
    w.setWindowTitle('Car trajectory with bounding boxes')

    rospy.init_node('reach_flow')
    my_node = Nodo(cw)