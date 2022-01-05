import pyredner
import torch
import redner
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh.transformations as transformations
import os
import h5py
import math
from matplotlib.pyplot import figure
from PIL import Image
from torchsummary import summary
import pandas as pd
import math

## euler from quaternion
import math
def radian_to_degree(x):
  return x * 180/math.pi

def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w, x, y, z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
      
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        
        return roll_x, pitch_y, yaw_z,  radian_to_degree(roll_x), radian_to_degree(pitch_y), radian_to_degree(yaw_z)# in radians



## translate poses from synthetic data (provided by Boewing) to "lookat Matirx" which can be used in rendering model
## t_xyz   translation coordinates xyz
## q_wxyz  quaternion w, x, y, z 
## return: camera lookat matrix  
def qt_to_lookat(t_xyz, q_wxyz):

  cm = transformations.quaternion_matrix(q_wxyz)
  
  #euler =transformations.euler_from_quaternion(q_wxyz)
  #pyredner.gen_rotate_matrix(euler)
  #print(cm)
  
  #construct camera matrix
  cm[:3,3] =  t_xyz
  # process difference of cooridinates representation between BLender and pyredner
  # blender : camera_up_vector: (0,0,1), axis order: x, y, z   pyrender: camera_up_vector (0,1,0)  axis order x, z, -y
  # in other words: a coordinates (a, b, c) in blender correspond (a, c, -b) in pyredner system
  #
  # switch axis y and axis z 
  # nagetive axis z, 
  cm[[1,2],:] = cm[[2,1],:]
  cm[2,:] = (-1)*cm[2,:]

  # negative vector "forward"
  cm[:,2] = (-1)*cm[:,2] 

  return cm

# convert euler angle to lookat matrix in Blender
def euler_to_lookat(t_xyz, euler_xyz):
  q_wxyz = transformations.quaternion_from_euler(euler_xyz[0],euler_xyz[1],euler_xyz[2])

  cm = qt_to_lookat(t_xyz, q_wxyz)

  return cm
## euler from quaternion

 
def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        def radian_to_degree(x):
          return x * 180/math.pi

        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
      
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        
        return [roll_x, pitch_y, yaw_z],  [radian_to_degree(roll_x), radian_to_degree(pitch_y), radian_to_degree(yaw_z) ]# in radians