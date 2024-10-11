#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 05 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque


##urdf location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)

weight_file = 'ddpg_meta_1.npy'
file_path = abspath(join(this_dir,weight_file))

data = np.load(file_path, allow_pickle=True)

print('con_eps: ', data[2])
print('con_step: ', data[3])