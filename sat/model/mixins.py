# -*- encoding: utf-8 -*-
"""
@File    :   mixins.py
@Time    :   2021/10/01 17:52:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
"""

import math
# here put the import lib
import os
import random
import sys

import torch

from .attention import *
from .base_model import BaseMixin
from .cached_autoregressive_model import CachedAutoregressiveMixin
from .finetune import *
