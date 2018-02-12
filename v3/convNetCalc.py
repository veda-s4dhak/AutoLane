# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:18:39 2018

@aut,hor: ,hP_OWNER
"""
import math

def computeOutDim(i, k, p, s):
    return(math.floor((i - k + 2*p)/s) + 1)
    
h = 300
w = 400

kw = [3,2,3,2,3,1,3,2,3,1,3,2,3,1,3,1,3,2,3,1,1,1,1,1,1,1,1,1]
kh = [3,2,3,2,3,1,3,2,3,1,3,2,3,1,3,1,3,2,3,1,1,1,1,1,1,1,1,1]

print(len(kw))
p = 0
s = [1,2,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


kw[19] = 3
kw[20] = 3
kw[21] = 3
kw[22] = 3
kw[23] = 3
kw[24] = 3
kw[25] = 2

ih = h
iw = w

for i in range(len(kw)):
      nh =   computeOutDim(ih, kh[i], p, s[i])
      nw =   computeOutDim(iw, kw[i], p, s[i])
      
      print('Layer ' ,i, ' ' ,nh, ' ', nw)
      
      ih = nh
      iw = nw


     
