# network of working memory using conjunctive-coding units
# Johanna Meyer
# based on Sanjay Manohar's code see http://www.smanohar.com/wp/wm/download.html

## general declarations
from numpy import *
from extract_features import extract_features
import matplotlib.pyplot as plt
from random import choice
from random import uniform


indices_array = load('indices_features.npy')
image_11=extract_features('images/test/test6/no_mooney1.jpg')
image_12=extract_features('images/test/test6/no_mooney2.jpg')
image_13=extract_features('images/test/test6/no_mooney3.jpg')
image_14=extract_features('images/test/test6/no_mooney4.jpg')
image_15=extract_features('images/test/test6/no_mooney5.jpg')
image_21=extract_features('images/test/test6/no_mooney6.jpg')
image_22=extract_features('images/test/test6/no_mooney7.jpg')
image_23=extract_features('images/test/test6/no_mooney8.jpg')
image_24=extract_features('images/test/test6/no_mooney9.jpg')
image_25=extract_features('images/test/test6/no_mooney10.jpg')
image_31=extract_features('images/test/test6/mooney1.jpg')
image_32=extract_features('images/test/test6/mooney2.jpg')
image_33=extract_features('images/test/test6/mooney3.jpg')
image_34=extract_features('images/test/test6/mooney4.jpg')
image_35=extract_features('images/test/test6/mooney5.jpg')
image_41=extract_features('images/test/test6/mooney6.jpg')
image_42=extract_features('images/test/test6/mooney7.jpg')
image_43=extract_features('images/test/test6/mooney8.jpg')
image_44=extract_features('images/test/test6/mooney9.jpg')
image_45=extract_features('images/test/test6/mooney10.jpg')


array_stimulus = concatenate((image_11[0].reshape(-1,1),image_12[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_13[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_14[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_15[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_21[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_22[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_23[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_24[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_25[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_31[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_32[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_33[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_34[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_35[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_41[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_42[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_43[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_44[0].reshape(-1,1)),axis=1)
array_stimulus = concatenate((array_stimulus,image_45[0].reshape(-1,1)),axis=1)


image_11_=extract_features('images/test/test6/no_mooney1_cue.jpg')
image_12_=extract_features('images/test/test6/no_mooney2_cue.jpg')
image_13_=extract_features('images/test/test6/no_mooney3_cue.jpg')
image_14_=extract_features('images/test/test6/no_mooney4_cue.jpg')
image_15_=extract_features('images/test/test6/no_mooney5_cue.jpg')
image_21_=extract_features('images/test/test6/no_mooney6_cue.jpg')
image_22_=extract_features('images/test/test6/no_mooney7_cue.jpg')
image_23_=extract_features('images/test/test6/no_mooney8_cue.jpg')
image_24_=extract_features('images/test/test6/no_mooney9_cue.jpg')
image_25_=extract_features('images/test/test6/no_mooney10_cue.jpg')
image_31_=extract_features('images/test/test6/mooney1_cue.jpg')
image_32_=extract_features('images/test/test6/mooney2_cue.jpg')
image_33_=extract_features('images/test/test6/mooney3_cue.jpg')
image_34_=extract_features('images/test/test6/mooney4_cue.jpg')
image_35_=extract_features('images/test/test6/mooney5_cue.jpg')
image_41_=extract_features('images/test/test6/mooney6_cue.jpg')
image_42_=extract_features('images/test/test6/mooney7_cue.jpg')
image_43_=extract_features('images/test/test6/mooney8_cue.jpg')
image_44_=extract_features('images/test/test6/mooney9_cue.jpg')
image_45_=extract_features('images/test/test6/mooney10_cue.jpg')


array_cue = concatenate((image_11_[0].reshape(-1,1),image_12_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_13_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_14_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_15_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_21_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_22_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_23_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_24_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_25_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_31_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_32_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_33_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_34_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_35_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_41_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_42_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_43_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_44_[0].reshape(-1,1)),axis=1)
array_cue = concatenate((array_cue,image_45_[0].reshape(-1,1)),axis=1)

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

  if len(stims)==1:

    I.durations = [     # duration for each stage of the input
    #250, 150,  375,  150, 275    
    #200,  120,50, 120,50, 120,50, 120,  300,   120, 220   
    #400, 240, 100, 240, 100, 240, 100, 240, 600, 240, 440 
    400, 500, 200, 340, 440
    ] 
    I.stimulus[2] = 0
    I.stimulus[4] = 0
    
  elif len(stims)==2:
    I.durations = [     # duration for each stage of the input
    #250, 150, 63, 150, 375,  150, 275    
    #200,  120,50, 120,50, 120,50, 120,  300,   120, 220   
    #400, 240, 100, 240, 100, 240, 100, 240, 600, 240, 440
    400, 500, 200, 500, 200, 340, 440 
    ]   
    I.stimulus[2] = 0
    I.stimulus[4] = 0
    I.stimulus[6] = 0
    
  elif len(stims)==3:
    I.durations = [     # duration for each stage of the input
    #250, 150, 63, 150, 63, 150, 375,  150, 275    
    #200,  120,50, 120,50, 120,50, 120,  300,   120, 220   
    #400, 240, 100, 240, 100, 240, 100, 240, 600, 240, 440
    400, 500, 200, 500, 200, 500, 200, 340, 440
    ] 
    I.stimulus[2] = 0
    I.stimulus[4] = 0
    I.stimulus[6] = 0
    I.stimulus[8] = 0
    
     
  elif len(stims)==4:
    I.durations = [     # duration for each stage of the input
    #250, 150, 63, 150, 63, 150, 63, 150,  375,  150, 275     
    #200,  120,50, 120,50, 120,50, 120,  300,   120, 220   
    #400, 240, 100, 240, 100, 240, 100, 240, 600, 240, 440
    400, 500, 200, 500, 200, 500, 200, 500, 200, 340, 440  
    ]
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

def single_trial(M,I,S,probe):
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
    
  ########################################################################   
  wh=sum(I.durations) - 150          
  final_array = S.F_t[wh]
  threshold_compare = 0.190
  
  final_array[final_array < threshold_compare] = 0
  final_array[final_array >= threshold_compare] = 1
  
    
  dot_array = zeros(20)
  i=0
  for i in range(20):
    compare = array_stimulus[:,i]
    compare[compare < threshold ] = 0
    compare[compare >= threshold ] = 1
    dot_array[i]= dot(final_array, compare)
    
  maxim = argmax(dot_array)
  if probe == maxim:
    isittrue = True 
  else:
    isittrue = False 

  return  isittrue,S  
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

################### Graph showing accuracy in relation to set size
def display_accuracy(accuracy,accuracy_mooney,error,error_mooney):
    
    
    plt.xlabel('Set size')
    plt.ylabel('Accuracy(%)')
    x = array([1,2,3,4])
    plt.axhline(y=1/5, xmin=0, xmax=5)

    plt.errorbar(x, accuracy,
             yerr = error,
             fmt ='o',label='no mooney faces')
    plt.errorbar(x, accuracy_mooney,
             yerr = error_mooney,
             fmt ='o',label='mooney faces')
    plt.xlim([0,5])
    plt.ylim([0,1])     
    plt.legend()    
    plt.show()         

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
    
  n_t = 20 # number of trials per number of presented items
  n_tr = 10 # number of test runs
  all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
  accuracy_=zeros([4,n_tr])
  accuracy_complex_=zeros([4,n_tr])
    
####### test for one presented items #######
  i=0
  while i<n_tr:

    right_answers = zeros(8)
    trials = zeros(8)

    print('1')
    j=0
    while j<n_t:
        a=random.choice(all)
        b=a
        (M,I) = setup_model_new([a],b)  # create Model and Input
        S     = initialise_state(M) # initialise the state of the units
        isittrue,S    = single_trial(M,I,S,b)
        if b < 10:
          trials[0] += 1
        else:
          trials[4] += 1
        if isittrue == True:
          if b < 10:
            right_answers[0] += 1
          else:
            right_answers[4] += 1
        j += 1 
        
####### test for two presented items #######
    print('2')  
    # get random number from all, find in which shape specific list the index also occurs, copy (rects_local=rects, remove idx from rects_local), take second random number from this
    j=0
    while j<n_t:
        a=random.choice(all)     
        b= random.choice(all)
        while b==a:
          b= random.choice(all)
        c=random.choice([a,b])
        (M,I) = setup_model_new([a,b],c)  # create Model and Input
        S     = initialise_state(M) # initialise the state of the units
        isittrue,S     = single_trial(M,I,S,c)
        #display_result(S)
        if c < 10:
          trials[1] += 1
        else:
          trials[5] += 1
        if isittrue == True:
          if c < 10:
            right_answers[1] += 1
          else:
            right_answers[5] += 1
        j += 1  
    
###### test for three presented items ######
    print('3')
    
    j=0
    while j<n_t:
        a=random.choice(all)     
        b= random.choice(all)
        while b==a:
          b= random.choice(all)
        c= random.choice(all)  
        while c==a or c==b:
          c= random.choice(all)        
        d=random.choice([a,b,c])
        (M,I) = setup_model_new([a,b,c],d)  # create Model and Input
        S     = initialise_state(M) # initialise the state of the units
        isittrue,S     = single_trial(M,I,S,d)
        if d < 10:
          trials[2] += 1
        else:
          trials[6] += 1
        if isittrue == True:
          if d < 10:
            right_answers[2] += 1
          else:
            right_answers[6] += 1
        j += 1 
    ######## test for four presented items (16 trials) ########
    print('4')
    j=0
    while j<n_t:

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
        (M,I) = setup_model_new([a,b,c,d],e)  # create Model and Input
        S     = initialise_state(M) # initialise the state of the units
        isittrue,S     = single_trial(M,I,S,e)
        if e< 10:
          trials[3] += 1
        else:
          trials[7] += 1
        if isittrue == True:
          if e < 10:
            right_answers[3] += 1
          else:
            right_answers[7] += 1

        j += 1 
    
    
  
    accuracy_[0,i]=right_answers[0]/trials[0]
    accuracy_[1,i]=right_answers[1]/trials[1]
    accuracy_[2,i]=right_answers[2]/trials[2]
    accuracy_[3,i]=right_answers[3]/trials[3]
    accuracy_complex_[0,i]=right_answers[4]/trials[4]
    accuracy_complex_[1,i]=right_answers[5]/trials[5]
    accuracy_complex_[2,i]=right_answers[6]/trials[6]
    accuracy_complex_[3,i]=right_answers[7]/trials[7]
    i +=1

  accuracy= mean(accuracy_,axis=1)
  accuracy_complex= mean(accuracy_complex_,axis=1)

  standard_error = zeros(4)
  standard_error_complex = zeros(4)
  
  standard_error[0] = std(accuracy_[0]) / sqrt(len(accuracy_[0])) 
  standard_error[1] = std(accuracy_[1]) / sqrt(len(accuracy_[1]))
  standard_error[2] = std(accuracy_[2]) / sqrt(len(accuracy_[2])) 
  standard_error[3] = std(accuracy_[3]) / sqrt(len(accuracy_[3]))
  standard_error_complex[0] = std(accuracy_complex_[0]) / sqrt(len(accuracy_complex_[0])) 
  standard_error_complex[1] = std(accuracy_complex_[1]) / sqrt(len(accuracy_complex_[1]))
  standard_error_complex[2] = std(accuracy_complex_[2]) / sqrt(len(accuracy_complex_[2])) 
  standard_error_complex[3] = std(accuracy_complex_[3]) / sqrt(len(accuracy_complex_[3]))
  display_accuracy(accuracy,accuracy_complex,standard_error,standard_error_complex)


def __main__():
  main_test_set_size()
result = __main__()