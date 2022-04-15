#Author Jay And Sunny

import math
import random

import string

class NN:
  def __init__(self, NI, NH, NO):
    # number of nodes in layers
    self.ni = NI + 1 # +1 for bias
    self.nh = NH + 1
    self.no = NO
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no

    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    # initialize node weights to random vals
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
    
  def runNN (self, inputs):
    # if len(inputs) != self.ni-1:
    #   print ('incorrect number of inputs')
    
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = sigmoid (sum)
    
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = sigmoid (sum)
      
    return self.ao
      
      
  
  def backPropagate (self, targets, N, M):
    
    output_deltas = [0.0] * self.no
    for k in range(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsigmoid(self.ao[k]) 
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        # print ('activation',self.ai[i],'synapse',i,j,'change',change)
        self.wi[i][j] += N*change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * (targets[k]-self.ao[k])**2
    return error
        
        
  
  def test(self, patterns):
    for p in patterns:
      inputs = p[0]
      print ('Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1])
      # print 'Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1]  
  def train (self, patterns, max_iterations = 3000, N=0.15, M=0.1):
    for i in range(max_iterations):
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.runNN(inputs)
        error = self.backPropagate(targets, N, M)
      if i % 50 == 0:
        print ('Combined error', error)
        # print 'Combined error', error
    self.test(patterns)
    

def sigmoid (x):
  return math.tanh(x)
  
def dsigmoid (y):
  return 1 - y**2

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)

def main ():
  pat = [
      [[-0.8,-0.1,-0.5,0.4], [1]],
      [[0.8,0.1,-0.5,0.4], [-1]],
      [[0.8,0.1,-0.5,-0.4], [-1]],
      [[-0.9,-0.2,-0.6,0.6], [1]]
  ]
  pat = [[[0.00653374610199271, 0.016167063327991272, 0.0011947402545964382, 0.06924329982967914], [0.7260891942236418]], [[0.004242752383655591, 0.05108989673912291, 0.09823922140520598, 0.05112211999922492], [0.7090541782361182]], [[0.010389152088657684, 0.06539657936133524, 0.04354359167877017, 0.051842812352558304], [0.740569300021157]], [[0.011917636185079029, 0.08903297990069049, 0.07336299002387718, 0.061048529542975995], [0.7227802497322611]], [[0.011864742517026294, 0.07785396316377857, 0.07883210155903889, 0.0004066028045362047], [0.7906141303097785]], [[0.046517120140843776, 0.02809747863551858, 0.0076189278588620684, 0.029421944422221027], [0.8131819708094595]], [[0.08036527389864978, 0.0817807163618936, 0.05496666255671695, 0.02735388258219278], [0.8386148274036567]], [[0.09059150477908479, 0.011844210803571155, 0.04690394888622477, 0.0031865863130955407], [0.8668726617675412]], [[0.05448069777954967, 0.059259654560134656, 0.09245012991257466, 0.013879959527458975], [0.8079626965293437]], [[0.01897702299741727, 0.07423920552265105, 0.07504147105718036, 0.010060755345755457], [0.7871355127906354]]]
  # 5 input layers, 10 Hidden layers, 3 output layers

  # myNN = NN ( 5, 10, 3)
  myNN = NN ( 4, 30, 1)
  myNN.train(pat)
  
  





if __name__ == "__main__":
    main()
