#accuracy_test.py
# network of working memory using conjunctive-coding units
# Johanna Meyer
# based on Sanjay Manohar's code

## general declarations
from numpy import *
from extract_features import extract_features
from random import choice

image_11=extract_features('images/test/test1/circle_blue.jpg')
image_12=extract_features('images/test/test1/circle_green.jpg')
image_13=extract_features('images/test/test1/circle_red.jpg')
image_14=extract_features('images/test/test1/circle_white.jpg')
image_15=extract_features('images/test/test1/circle_yellow.jpg')

array_stimulus = concatenate((image_11[0].reshape(-1,1),image_12[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_13[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_14[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_15[0].reshape(-1,1)),axis=1)


image_11_=extract_features('images/test/test1/circle_blue_cue.jpg')
image_12_=extract_features('images/test/test1/circle_green_cue.jpg')
image_13_=extract_features('images/test/test1/circle_red_cue.jpg')
image_14_=extract_features('images/test/test1/circle_white_cue.jpg')
image_15_=extract_features('images/test/test1/circle_yellow_cue.jpg')


array_cue = concatenate((image_11_[0].reshape(-1,1),image_12_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_13_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_14_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_15_[0].reshape(-1,1)),axis=1)

threshold = 1.55 * mean(array_stimulus)



NF        = len(image_11[0])     # number of feature units
def setup_model_new(stims, probe):
  ## Model definition
  M = Model()
  M.NF        = NF
  M.NC        = 4
  M.baselineC = 0.2  # level to which conjunctive units decay
  M.baselineF = 0.2  # level to which feature units decay
  M.decayC    = 1.0  # decay of activity per timestep (1 = no decay)
  M.inhibC    = 0.5  # mutual inhibition between conjunctive units
  M.learnC    = 0.01  # learning rate for CF weights
  M.learnF    = 0.005 #               for FC weights
  M.gainfC    = 0.96 /M.NF
  M.gaincF    = 1.2 /M.NF
  M.noiseC    = 0.005 # noise in conjunction units
  M.noiseWC   = 0     # noise in FC weights
  M.inhibF    = 0.3
  M.inhibFf   = 0.20
  M.decayF    = 0.69 
  M.sigmoid = lambda x: maximum(0,minimum(1,x))
  
  ## Input Condition definition
  I = Input()
  I.durations = [     # duration for each stage of the input
    #250, 150, 63, 150, 63, 150, 63, 150,  375,  150, 275     
    #200,  120,50, 120,50, 120,50, 120,  300,   120, 220   
    400, 500, 200, 500, 200, 500, 200, 500, 200, 340, 440  
    ]  
  
  # Stages of input to the feature units (each row is one stage)
  # Each F unit receives input to keep it at the given value.
  # NaN means no external input to the network
  black_screen=ones((M.NF), dtype=int)*-1

  I.stimulus = black_screen  
  for stim in stims:
      I.stimulus = row_stack((I.stimulus, array_stimulus[:,stim], zeros(M.NF)))

  I.stimulus = row_stack((I.stimulus,array_cue[:,probe]))
  I.stimulus = row_stack((I.stimulus,black_screen))
  I.stimulus[I.stimulus < threshold ] = -1
  I.stimulus[I.stimulus >= threshold] = 1
  I.stimulus[2] = 0
  I.stimulus[4] = 0
  I.stimulus[6] = 0
  I.stimulus[8] = 0
  I.stimulus[10] = 0
  

  return (M,I)

#################################################################

def do_one_timestep(S,M):
  # S = current model state
  # M = model parameters
  
  S.F = M.sigmoid(
    M.baselineF
    + M.decayF            * (S.F - M.baselineF) # decay
    + M.gaincF * S.Wcf    * (S.C - M.baselineC) # input from conj
    + M.inhibF * (S.F - M.baselineF) # f->f autoactivation
    - M.inhibFf * sum(S.F - M.baselineF)/len(S.F) #  feature to feature inhibition
    + S.stim
  )

  # 2. Update Conjunctive units 
  S.C = M.sigmoid(
    M.baselineC 
    + M.decayC            * (S.C - M.baselineC) # decay to baseline
    + M.gainfC * S.Wfc    * (S.F - M.baselineF) # input from features
    - M.inhibC * sum(S.C - M.baselineC) # lateral inhibitio
    + M.noiseC * random.normal(size=S.C.shape)  # noise
  )
  # 3. Change in weights
  dW    = (S.C - M.baselineC) * (S.F.T - M.baselineF)
  S.Wfc = M.sigmoid(
    S.Wfc + M.learnF * dW
          + M.noiseWC * random.normal(size=dW.shape)
  )
  S.Wcf = M.sigmoid(
    S.Wcf + M.learnC * dW.T
  )
  return S
  
#################################################################

def single_trial(M,I,S):
  # run a single trial, with model M, inputs I,
  # and initial state S
  time  = 0
  stage = 0
  stage_end_times = cumsum(I.durations) # when does each stage end?
  # create nonlinear transfer function 
  


  ended = False
  # create time series for storing conjunctive and feature unit activity
  F_t = nan*zeros( (stage_end_times[-1],  S.F.size) )
  C_t = nan*zeros( (stage_end_times[-1],  S.C.size) )

  # start trial with zero activity
  S.C[:] = 0
  S.F[:] = 0
  
  while not ended:
    
    # 1. Get current Input stimulus
    time += 1
    if time > stage_end_times[stage]:
      stage += 1     
    if stage >= stage_end_times.size:  
      ended=True
      break
    

    S.stim = I.stimulus[(stage),:][:,None]   # select a row, convert to column
    
    
    # 2. Run one timestep
    S = do_one_timestep(S,M)
    
    F_t[time-1,:] = S.F.T        # store current activity
    C_t[time-1,:] = S.C.T
    if time==3540:
        S.F[:] = 0
    if time==3200:
      M.learnC=0
      M.learnF=0
    if time == sum(I.durations) - 250:    
      wh=sum(I.durations) - 250    
      final_array = S.F[:]
      threshold_compare = 0.195
  
      final_array[final_array < threshold_compare] = 0
      final_array[final_array >= threshold_compare] = 1  
      S.F[:] = final_array 


  S.F_t = F_t     # finished one trial - save activity timecourse.
  S.C_t = C_t
  
  # what was the response? look at final feature values
  win_feature = where( S.F[4:8] == S.F[4:8].max() )[0]
  if win_feature.size > 1:                     # multiple maxima found
    S.win_feature = random.choice(win_feature) # tie. randomly select one
  elif win_feature.size == 0:                  # no maximum found
    S.win_feature = nan                        # (oh dear)
  else:                                        # just one maximum found
    S.win_feature = win_feature[0]             # (oh good)

  wh=sum(I.durations) - 300         
  final_array = S.F_t[wh]
  
  print(mean(final_array))  
  
    
  return  S

############################################################

def initialise_state(M):
  # Initialise states (rates and weights)  of units.
  # return an object with F, C, wcij and wfij.
  s = State()
  # 1. Firing rates of Feature units
  s.F = matrix(zeros(M.NF)).T
  # 2. Firing rates of Conjunction units
  s.C = asmatrix( zeros((M.NC,1)) )
  # 3. Connection weights from features to conjunctions
  s.Wfc = asmatrix(random.normal(size=(s.C.size,s.F.size))) 
  # 4. Connection weights from conjunctions to features
  s.Wcf = asmatrix(random.normal(size=(s.F.size,s.C.size)))
  return s

  


#################### Classes for holding structured data
class Model(object):
  pass

class State(object):
  pass

class Input(object):
  pass    

#################### Entry point

def main_test_set_size():       
  # run a single trial and display graph
    
    all = [0,1,2,3,4]
    
    ######## test for four presented items (16 trials) ########


    a=random.choice(all)      
    b= random.choice(all)
    while b==a:
      b= random.choice(all)
    c= random.choice(all)  
    while c==a or c==b:
      c= random.choice(all)   
    d= random.choice(all)  
    while d==a or d==b or d ==c:
      d= random.choice(all)   
    e=random.choice([a,b,c,d])
    print(a,b,c,d,e)
    (M,I) = setup_model_new([a,b,c,d],e)  # create Model and Input
    S     = initialise_state(M) # initialise the state of the units
    S     = single_trial(M,I,S,)
    display_result(S)


    
 

def display_result(S): 
  import matplotlib.pyplot as plt  
  import time  
  f=plt.figure(1)  
  plt.subplot(2,1,1)  
  plt.imshow(S.F_t.T,aspect='auto',interpolation='none')  
  plt.subplot(2,1,2)  
  plt.plot(S.C_t)  
  plt.xlim((0,S.C_t.shape[0]))  
  plt.show()  
  f.canvas.draw()  
  time.sleep(1e-2)



def __main__():
  main_test_set_size()
  print('finish')
result = __main__()
