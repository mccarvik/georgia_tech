""" 			  		 			     			  	   		   	  			  	
Utiliy functions.  (c) 2024 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""
import time, os

from models.cnn import hello_do_you_copy as cnn_hello_do_you_copy
from models.my_model import hello_do_you_copy as mm_hello_do_you_copy
from models.resnet import hello_do_you_copy as res_hello_do_you_copy
from models.twolayer import hello_do_you_copy as tl_hello_do_you_copy
from losses.focal_loss import hello_do_you_copy as fc_hello_do_you_copy


def say_hello_do_you_copy(drive_path: str) -> None:
  print('---------- Models ------------------')
  cnn_hello_do_you_copy()
  mm_hello_do_you_copy()
  res_hello_do_you_copy()
  tl_hello_do_you_copy()


  cnn_path = os.path.join(drive_path, 'models', 'cnn.py')
  mm_path = os.path.join(drive_path, 'models', 'my_model.py')
  res_path = os.path.join(drive_path, 'models', 'resnet.py')
  tl_path = os.path.join(drive_path, 'models', 'twolayer.py')

  cnn_edit_time = time.ctime(os.path.getmtime(cnn_path))
  mm_edit_time = time.ctime(os.path.getmtime(mm_path))
  res_edit_time = time.ctime(os.path.getmtime(res_path))
  tl_edit_time = time.ctime(os.path.getmtime(tl_path))


  print('cnn.py last edited on %s' % cnn_edit_time)
  print('my_model.py last edited on %s' % mm_edit_time)
  print('resnet.py last edited on %s' % res_edit_time)
  print('twolayer.py last edited on %s' % tl_edit_time)

  print()
  print('---------- Losses ------------------')

  fc_hello_do_you_copy()

  fc_path = os.path.join(drive_path, 'losses', 'focal_loss.py')
  fc_edit_time = time.ctime(os.path.getmtime(fc_path))
  print('focal_loss.py last edited on %s' % fc_edit_time)
