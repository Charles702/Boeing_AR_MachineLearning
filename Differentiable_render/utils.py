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
import sys
from subprocess import call

#make video of optimization process
def make_video_by_images(save_dir, name, format, framerate):
  # framerate is type of string
    call(["ffmpeg", "-framerate", framerate, "-i",
      save_dir + "{}_%d.{}".format(name,format), "-vb", "30M",
      save_dir+ "{}.avi".format(name)])
      
# convert string to class 
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


# Generate poses arround target position
def intoplate_poses(t_p, t_fname, sample_n, dev_p=[4, 4, 4], dev_r = [0.3, 0.3, 0.3]):
    t_position = t_p[:3]
    t_quaternion = t_p[3:]

    # convert quaternion to urler for easiler interpolation around target rotation
    # _, t_euler = euler_from_quaternion(t_quaternion)
    t_euler = transformations.euler_from_quaternion(t_quaternion)

    #sample_p: keep rotation fixed, change position only
    #sample_r: keep position fixed, change rotation only
    #smaple_pr: change position and rotation
    sample_p, sample_r, sample_pr = sample_n

    # deviation form x, y, z
    #devi_p = [0.5, 0.5, 0.5]
    #devi_r = [10, 10, 10]
    
    s_group = []
    # set target pos as the first element
 
    # choice_p = np.linspace(-1 * dev_p[0], dev_p[0], 51)
    # choice_r = np.linspace(-1 * dev_r[1], dev_r[2], 51)

    #generate samples using normal distribution
    dev_p_x = make_normal_bounded(-dev_p[0], +dev_p[0], sigma=dev_p[0]*.2, nsamples=200)
    dev_p_y = make_normal_bounded(-dev_p[1], +dev_p[1], sigma=dev_p[1]*.2, nsamples=200)
    dev_p_z = make_normal_bounded(-dev_p[2], +dev_p[2], sigma=dev_p[2]*.2, nsamples=200)

    dev_r_x = make_normal_bounded(-dev_r[0], +dev_r[0], sigma=dev_r[0]*.2, nsamples=200)
    dev_r_y = make_normal_bounded(-dev_r[1], +dev_r[1], sigma=dev_r[1]*.2, nsamples=200)
    dev_r_z = make_normal_bounded(-dev_r[2], +dev_r[2], sigma=dev_r[2]*.2, nsamples=200)

    #group.appen(t_p)    
    for i in range(sample_p):
      # new position x , y, z
      # np.random.choice assume uniform distribution for th list
      p_x = np.random.choice(dev_p_x) + t_position[0]
      p_y = np.random.choice(dev_p_y) + t_position[1]
      p_z = np.random.choice(dev_p_z) + t_position[2]
      # create file name for source images: _sp indicates: source images, only change position respect to target
      s_fname = t_fname[:-4] + '_sp'+ str(i).zfill(3) +'.jpg'
      s_pose = [s_fname] + [p_x, p_y, p_z] + t_quaternion
      s_group.append(s_pose)

    for i in range(sample_r):
      # new rotation: eular degree, x, y, z
      r_x = np.random.choice(dev_r_x) + t_euler[0]
      r_y = np.random.choice(dev_r_y) + t_euler[1]
      r_z = np.random.choice(dev_r_z) + t_euler[2]
      # convert euler back to quaternion
      # create file name for source images: _sr indicates: source images, only rotation position respect to target
      s_fname = t_fname[:-4] + '_sr'+ str(i).zfill(3)+ '.jpg'
      n_s_q = transformations.quaternion_from_euler(r_x, r_y, r_z)
      s_pose = [s_fname] + t_position + list(n_s_q)
      s_group.append(s_pose)

    for i in range(sample_pr):
      # new position x , y, z
      p_x = np.random.choice(dev_r_x) + t_position[0]
      p_y = np.random.choice(dev_r_y) + t_position[1]
      p_z = np.random.choice(dev_r_z) + t_position[2]

      # new rotation: eular degree, x, y, z
      r_x = np.random.choice(dev_r_x) + t_euler[0]
      r_y = np.random.choice(dev_r_y) + t_euler[1]
      r_z = np.random.choice(dev_r_z) + t_euler[2]
      # convert euler back to quaternion
      # create file name for source images: _spr indicates: source images, change position and rotation
      s_fname = t_fname[:-4] + '_spr'+ str(i).zfill(3)+ '.jpg'
      n_s_q = transformations.quaternion_from_euler(r_x, r_y, r_z)
      s_pose = [s_fname] + [p_x, p_y, p_z] + list(n_s_q)
      s_group.append(s_pose)

    print(len(s_group))
    return s_group
      
def legency_gen_data(): #----- temporarily deprecated 
    print("don't go here")
    #genderate source image names
    #position offset along x axis:  positive  and negative
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spx_p{st:03}.jpg", np.arange(len(offset_px_postive))))
    s_orignal[0,:] += offset_px_postive
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spx_n{st:03}.jpg", np.arange(len(offset_py_negative))))
    s_orignal[0,:] += offset_px_negative
    add_group(s_orignal, s_fname)    

    #position offset along y axis: positive  and negative
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spy_p{st:03}.jpg", np.arange(len(offset_py_postive))))
    s_orignal[1,:] += offset_py_postive
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spy_n{st:03}.jpg", np.arange(len(offset_py_negative))))
    s_orignal[1,:] += offset_py_negative
    add_group(s_orignal, s_fname)

    #oposition offset along z axis
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spz_p{st:03}.jpg", np.arange(len(offset_pz_postive))))
    s_orignal[2,:] += offset_pz_postive
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spz_n{st:03}.jpg", np.arange(len(offset_pz_negative))))
    s_orignal[2,:] += offset_pz_negative
    add_group(s_orignal, s_fname)

    #offset along xyz positive
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spxyz_p{st:03}.jpg", np.arange(len(offset_pz_postive))))
    s_orignal[0,:] += offset_px_postive 
    s_orignal[1,:] += offset_py_postive 
    s_orignal[2,:] += offset_pz_postive 
    add_group(s_orignal, s_fname)   

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_spxyz_n{st:03}.jpg", np.arange(len(offset_pz_postive))))
    s_orignal[0,:] += offset_px_negative
    s_orignal[1,:] += offset_py_negative
    s_orignal[2,:] += offset_pz_negative
    add_group(s_orignal, s_fname)     

    #--------------------------------------
    # rotation offset around x axis
    ##roation offset is based on euler angle
    def quaternion_offset(tg_euler, offset_r_euler, mode):
    # return quaterion given target euler angle and a list of offset euler angles
      q_after_offset = []
      for e in offset_r_euler:
        if mode == 'x': # euler offset around x
          s_q = transformations.quaternion_from_euler(tg_euler[0] + e, tg_euler[1], tg_euler[2])
          q_after_offset.append(s_q)
        elif mode == 'y':
          s_q = transformations.quaternion_from_euler(tg_euler[0] , tg_euler[1]+ e, tg_euler[2])
          q_after_offset.append(s_q)
        elif mode == 'z':
          s_q = transformations.quaternion_from_euler(tg_euler[0] , tg_euler[1], tg_euler[2]+ e)
          q_after_offset.append(s_q)
      return q_after_offset
      
    s_all = np.array(t_p).reshape(7,1) * np.ones((1, sample_r))
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srx_p{st:03}.jpg", np.arange(len(offset_rx_postive))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler,offset_rx_postive,'x' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srx_n{st:03}.jpg", np.arange(len(offset_rx_negative))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler, offset_rx_negative, 'x' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)   

    # rotation offset around y axis
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_sry_p{st:03}.jpg", np.arange(len(offset_ry_postive))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler, offset_ry_postive, 'y' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_sry_n{st:03}.jpg", np.arange(len(offset_ry_negative))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler, offset_ry_negative, 'y' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)

    # rotation offset around z axis
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srz_p{st:03}.jpg", np.arange(len(offset_rz_postive))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler, offset_rz_postive, 'z' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srz_n{st:03}.jpg", np.arange(len(offset_rz_negative))))
    # offset based on euler , need to be converted back to queaternion
    offset_q = quaternion_offset(t_euler, offset_rz_negative, 'z' )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)   

    #rotation offset around xyz axis - positive
    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srxyz_p{st:03}.jpg", np.arange(len(offset_rz_negative))))
    offset_xyz_positive = [offset_rx_postive, offset_ry_postive, offset_rz_postive]
    offset_q = quaternion_offset_xyz(t_euler, offset_xyz_positive )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname)   

    s_orignal = s_all.copy()
    s_fname = list(map(lambda st: f"{t_fname[:-4]}_srxyz_n{st:03}.jpg", np.arange(len(offset_rz_negative))))
    offset_xyz_negative = [offset_rx_negative, offset_ry_negative, offset_rz_negative]
    offset_q = quaternion_offset_xyz(t_euler, offset_xyz_negative )
    s_orignal[3:,:] = np.array(offset_q).T #transpose to fit the shape
    add_group(s_orignal, s_fname) 

import random
#prepare image data and annotation
def gen_groups(meta_dir, file_name, dir_s, cam_resolution, sample_n = [10,10,10]):
    # generate new metadata file: 
    # meta_dir: where original meta text stored
    # "dir_s", directory where source images stored
    # sample_n: number of samples generated, sample_n[0]:number of samples with changed position, 
    # sample_n[1]: number of sample with chaged rotation, sample_n[2]: number of sample with chaged position and rotation, 
   
    # 0. read metadata from txt
    new_meta_fname = 'metadata_pairs.txt'
    metadata_dir = os.path.join(meta_dir, file_name)
    new_meta_dir = os.path.join(meta_dir, new_meta_fname)

    if os.path.exists(new_meta_dir):
      os.remove(new_meta_dir)

    with open(os.path.join(metadata_dir),'r') as f, open(os.path.join(new_meta_dir),'a') as n_meta_f:
        #next(f)
        #next(f)
        #next(f)
        count = 0      # number of target images 
        for line in f:
          count += 1
          fname, p0, p1, p2, p3, p4, p5, p6, _ = line.split()
          #rename the file
          idx = fname.find('rgb_')
          t_fname = fname[idx:]
          #print("------",t_fname)
          
          #generate pose around target:
          t_p = [float(p0), float(p1), float(p2), float(p3), float(p4), float(p5), float(p6)]
          
          #oganize a set of poses around target
          s_group = intoplate_poses(t_p, t_fname, sample_n)

          #generate training metadata and render images 
          #training metadata consist of pairs of target image and source image
          lines = []
          for data in s_group:
          ## organize the pairs
             pose_t = p0+'|'+p1+'|'+p2+'|'+p3+'|'+p4+'|'+p5+ '|'+p6
             pose_s = str(data[1]) + '|'+str(data[2])+'|'+str(data[3])+'|'+str(data[4])+'|'+str(data[5])+'|'+str(data[6])+ '|'+str(data[7])
             name_s = data[0]
             line =  t_fname + ' ' + name_s + ' ' + pose_t + ' ' + pose_s + '\n'
             lines.append(line)

             # render source image
             
             render_sources(data, dir_s, cam_resolution)
             
          
          # save to file
          # form in new metadata fiel:  target_image_name, source_image_name, target_pose, source_pose
          n_meta_f.writelines(lines)
          print('--- ',t_fname, "  processed")

    #calculate the size of dataset
    dst_size = count*sum(sample_n)

    return dst_size, count

import cv2
# convert datset to h5 format
# shrink size of h5 dataset by removing repeated target images
def gen_dataseth5_shrink(training_t_dir, training_s_dir , dir_h5, meta_name, resolution, dst_size, tg_size,  greyScale=False, applySeg=True):
    # create h5 file
    c = 1 if greyScale else 3
    h = resolution[0]
    w = resolution[1]
    trainval_n = 'trainval_learn_operator_compress.h5'
    fileh5_p = os.path.join(dir_h5, trainval_n) 

    meta_p = os.path.join(training_t_dir,meta_name)
    with h5py.File(fileh5_p,'w') as f_dst_h5, open(meta_p,'r') as f_ann:
        print('records size ----', dst_size)
        #contruct h5 file
       
        dst_im_t = f_dst_h5.create_dataset('Train_im_t', (tg_size,h,w,c), 'f')
        dst_im_s = f_dst_h5.create_dataset('Train_im_s', (dst_size,h,w,c), 'f')
        pose_m1 = f_dst_h5.create_dataset('Train_p_t', (dst_size,7), 'f')
        pose_m2 = f_dst_h5.create_dataset('Train_p_s', (dst_size,7), 'f')

        for i, line in enumerate(f_ann):
          img_pair = line.strip().split()
          #print(img_pair)
          # targe img 
          t_img_name =  img_pair[0][4:-4].zfill(5)+'.jpg'
          img_t_path =  os.path.join(training_t_dir, 'JPEGImages',t_img_name)
          #seg img
          seg_name = img_pair[0][4:-4].zfill(5) + '.png'
          img_seg_path =  os.path.join(training_t_dir, 'SegmentationClassRaw',seg_name) 
          #source img
          img_s_path =  os.path.join(training_s_dir, img_pair[1])

          t_pose = img_pair[2].strip().split("|")
          s_pose = img_pair[3].strip().split("|")
          t = [float(t_pose[0]), float(t_pose[1]),float(t_pose[2]),float(t_pose[3]),float(t_pose[4]), float(t_pose[5]), float(t_pose[6])]
          s = [float(s_pose[0]), float(s_pose[1]),float(s_pose[2]),float(s_pose[3]),float(s_pose[4]), float(s_pose[5]), float(s_pose[6])]
          #pose_labels.append([t, s])

          #read imgs
          img_t = Image.open(img_t_path)
          img_s = Image.open(img_s_path)
          img_seg = cv2.imread(img_seg_path)
          #print('t_size--',img_t.size) #t_size-- (513, 289)
          #print('s_size--',img_s.size) #s_size-- (455, 256)
          
          # print('---', self.pose_labels[index][0])
          # convert to greyscale ?
          if greyScale:
            img_t = np.asarray(img_t.convert("L"))
            img_s = np.asarray(img_s.convert("L"))
            # apply segmentation on target image
            if applySeg:
              img_t = img_t * img_seg[0]
          else:
            img_t = np.asarray(img_t)
            img_s = np.asarray(img_s)
            if applySeg:
              img_t = img_t * img_seg

          #fill dataset:
          # one target img coresponds to multiple source imgs,  how to reduce the redundency?
          number_sampel = int(dst_size/tg_size)
          dst_im_t[i//number_sampel] = img_t
          print('tg--',i//number_sampel)

          dst_im_s[i] = img_s
          pose_m1[i] = t
          pose_m2[i] = s
          print(f'%d processed'% i )


import cv2
# convert datset to h5 format
def gen_dataseth5(training_t_dir, training_s_dir , dir_h5, meta_name, resolution, dst_size, greyScale=False, applySeg=True):
    # create h5 file
    c = 1 if greyScale else 3
    h = resolution[0]
    w = resolution[1]
    trainval_n = 'trainval_learn_operator.h5'
    fileh5_p = os.path.join(dir_h5, trainval_n) 

    meta_p = os.path.join(training_t_dir,meta_name)
    with h5py.File(fileh5_p,'w') as f_dst_h5, open(meta_p,'r') as f_ann:
        print('records size ----', dst_size)
        #contruct h5 file
       
        dst_im_t = f_dst_h5.create_dataset('Train_im_t', (dst_size,h,w,c), 'f')
        dst_im_s = f_dst_h5.create_dataset('Train_im_s', (dst_size,h,w,c), 'f')
        pose_m1 = f_dst_h5.create_dataset('Train_p_t', (dst_size,7), 'f')
        pose_m2 = f_dst_h5.create_dataset('Train_p_s', (dst_size,7), 'f')

        for i, line in enumerate(f_ann):
          img_pair = line.strip().split()
          #print(img_pair)
          # targe img 
          t_img_name =  img_pair[0][4:-4].zfill(5)+'.jpg'
          img_t_path =  os.path.join(training_t_dir, 'JPEGImages',t_img_name)
          #seg img
          seg_name = img_pair[0][4:-4].zfill(5) + '.png'
          img_seg_path =  os.path.join(training_t_dir, 'SegmentationClassRaw',seg_name) 
          #source img
          img_s_path =  os.path.join(training_s_dir, img_pair[1])

          t_pose = img_pair[2].strip().split("|")
          s_pose = img_pair[3].strip().split("|")
          t = [float(t_pose[0]), float(t_pose[1]),float(t_pose[2]),float(t_pose[3]),float(t_pose[4]), float(t_pose[5]), float(t_pose[6])]
          s = [float(s_pose[0]), float(s_pose[1]),float(s_pose[2]),float(s_pose[3]),float(s_pose[4]), float(s_pose[5]), float(s_pose[6])]
          #pose_labels.append([t, s])

          #read imgs
          img_t = Image.open(img_t_path)
          img_s = Image.open(img_s_path)
          img_seg = cv2.imread(img_seg_path)
          #print('t_size--',img_t.size) #t_size-- (513, 289)
          #print('s_size--',img_s.size) #s_size-- (455, 256)
          
          # print('---', self.pose_labels[index][0])
          # convert to greyscale ?
          if greyScale:
            img_t = np.asarray(img_t.convert("L"))
            img_s = np.asarray(img_s.convert("L"))
            # apply segmentation on target image
            if applySeg:
              img_t = img_t * img_seg[0]
          else:
            img_t = np.asarray(img_t)
            img_s = np.asarray(img_s)
            if applySeg:
              img_t = img_t * img_seg

          #fill dataset:
          # one target img coresponds to multiple source imgs,  how to reduce the redundency?
          dst_im_t[i] = img_t
          dst_im_s[i] = img_s
          pose_m1[i] = t
          pose_m2[i] = s
          print(f'%d processed'% i )


def train_on_pairs():
  for epoch in range(epochs):
    
    for i, data in enumerate(train_dataloader, 0):
      img_t, img_s, pose_t, pose_s = data 

      img_t = img_t.cuda()
      img_s = img_s.cuda()
      pose_t = pose_t.cuda()
      pose_s = pose_s.cuda()

      optimizer.zero_grad()
      output = net(img_t, img_s)
      #out_t, out_s = net(img_t, img_s)

      # calculate the distance between two pose, 
      # "dif_pos" will be used as labels for convergence of network output
      dif_pos  =  cal_pose_dis(pose_t, pose_s)

      # calculate the loss
      #loss = criterion(out_t, out_s, dif_pos)
      loss = criterion(output, dif_pos)

      loss.backward()
      optimizer.step()
    

    t = time.time()-start
    print("Time:{}, Loss of {} epoch:  {}".format(t, epoch+1, loss.item()))
      
    ## validation 
    ## size t : s = 10:300 
      

  ## save the model
  torch.save(net.state_dict(), './Learn_operators/model.pt')
  print("save succesfully") 

   #train_on_pairs()

def application_h5():
    ##  H5 file
    import numpy as np
    import h5py
    from PIL import Image  



    fileName = 'data.h5'
    numOfSamples = 5
    with h5py.File(fileName, "w") as out:
      out.create_dataset("X_train",(numOfSamples,256,256,3),dtype='u1')
      out.create_dataset("Y_train",(numOfSamples,1,1),dtype='u1')      
      out.create_dataset("X_dev",(numOfSamples,256,256,3),dtype='u1')
      out.create_dataset("Y_dev",(numOfSamples,1,1),dtype='u1')      
      out.create_dataset("X_test",(numOfSamples,256,256,3),dtype='u1')
      out.create_dataset("Y_test",(numOfSamples,1,1),dtype='u1')   

    test_name = 'test.h5'
    with h5py.File(test_name, 'w') as f:
      ds = f.create_dataset('X_train_m1',(2,5,5), 'f')
      for i in range(3):
        row = np.random.rand(5,5)
        ds[i] = row
      print(ds[0,0,0])
      #f.create_dataset('X_train_m2',(5,5,5), 'f')
      labels = f.create_dataset('Y_train_m1',(2,), 'f')
      for i in range(3):
        lab = np.random.random(3)
        labels[i] = lab
      # f.create_dataset('Y_train_m2',(5,3), 'f')

