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

from modules.conv_classifier import hello_do_you_copy as cc_hello_do_you_copy
from modules.softmax_ce import hello_do_you_copy as sce_hello_do_you_copy
from modules.linear import hello_do_you_copy as ln_hello_do_you_copy
from modules.relu import hello_do_you_copy as rl_hello_do_you_copy
from modules.max_pool import hello_do_you_copy as ml_hello_do_you_copy
from modules.convolution import hello_do_you_copy as cv_hello_do_you_copy
from optimizer._base_optimizer import hello_do_you_copy as bo_hello_do_you_copy
from optimizer.sgd import hello_do_you_copy as sgd_hello_do_you_copy

def say_hello_do_you_copy(drive_path: str) -> None:
  print('---------- Modules ------------------')
  cc_hello_do_you_copy()
  sce_hello_do_you_copy()
  ln_hello_do_you_copy()
  rl_hello_do_you_copy()
  ml_hello_do_you_copy()
  cv_hello_do_you_copy()


  cc_path = os.path.join(drive_path, 'modules', 'conv_classifier.py')
  sce_path = os.path.join(drive_path, 'modules', 'softmax_ce.py')
  ln_path = os.path.join(drive_path, 'modules', 'linear.py')
  rl_path = os.path.join(drive_path, 'modules', 'relu.py')
  ml_path = os.path.join(drive_path, 'modules', 'max_pool.py')
  cv_path = os.path.join(drive_path, 'modules', 'convolution.py')

  cc_edit_time = time.ctime(os.path.getmtime(cc_path))
  sce_edit_time = time.ctime(os.path.getmtime(sce_path))
  ln_edit_time = time.ctime(os.path.getmtime(ln_path))
  rl_edit_time = time.ctime(os.path.getmtime(rl_path))
  ml_edit_time = time.ctime(os.path.getmtime(ml_path))
  cv_edit_time = time.ctime(os.path.getmtime(cv_path))


  print('conv_classifier.py last edited on %s' % cc_edit_time)
  print('softmax_ce.py last edited on %s' % sce_edit_time)
  print('linear.py last edited on %s' % ln_edit_time)
  print('relu.py last edited on %s' % rl_edit_time)
  print('max_pool.py last edited on %s' % ml_edit_time)
  print('convolution.py last edited on %s' % cv_edit_time)

  print()
  print('---------- Optimizer ------------------')
  bo_hello_do_you_copy()
  sgd_hello_do_you_copy()

  bo_path = os.path.join(drive_path, 'optimizer', '_base_optimizer.py')
  sgd_path = os.path.join(drive_path, 'optimizer', 'sgd.py')
  base_optimizer_edit_time = time.ctime(os.path.getmtime(bo_path))
  sgd_edit_time = time.ctime(os.path.getmtime(sgd_path))
  print('_base_optimizer.py last edited on %s' % base_optimizer_edit_time)
  print('sgd.py last edited on %s' % sgd_edit_time)
