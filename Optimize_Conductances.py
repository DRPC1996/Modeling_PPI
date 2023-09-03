import numpy as np
from deap import base, creator, tools, algorithms

from brian2 import *
from brian2tools import *

def stimuli(typeOfStimulus, ISI=5, plotFigure = False):
    
    s_amp = 100     # Startling amplitude
    pre_amp = 70    # Pre-pulse amplitude
    
    s_len = 10      # Startling pulse length in ms
    pre_len =10     # Pre-pulse lenght in ms
    
    if typeOfStimulus == 0:     # Startling Stimulus
        
        stimulus = TimedArray(array([0]*50+[s_amp]*s_len+[0]*50), dt=1*ms)
        to_plot_stimuli = array([0]*50+[s_amp]*s_len+[0]*50)
        
    elif typeOfStimulus == 1:   # Startiling Stimulus followed by a pre pulse
        
        stimulus = TimedArray(array([0]*50+[pre_amp]*pre_len+[0]*ISI+[s_amp]*s_len+[0]*50), dt=1*ms)
        to_plot_stimuli = array([0]*50+[pre_amp]*pre_len+[0]*ISI+[s_amp]*s_len+[0]*50)
    
    elif typeOfStimulus == 2:   # Only the pre-pulse
        
        stimulus = TimedArray(array([0]*50+[pre_amp]*pre_len+[0]*50), dt=1*ms)
        to_plot_stimuli = array([0]*50+[pre_amp]*pre_len+[0]*50) 
        
    if plotFigure:
        plt.plot(to_plot_stimuli)
        plt.show()
    
    return stimulus,to_plot_stimuli

# Utility functions

def find_sequence_ranges(arr):
    ranges = []
    start = 0

    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            ranges.append((start, i - 1))
            start = i

    ranges.append((start, len(arr) - 1))
    return ranges

def change_range_values(arr, start, end, new_value):
    for i in range(start, end + 1):
        arr[i] = new_value


def save_array_to_file(filename, array):
    with open(filename, 'a') as file:
        for element in array:
            file.write(str(element) + ',')
        file.write('\n')

def evaluate(individual):
        
    simulated_output = simulation(individual)
    
    error = np.mean((simulated_output - reference_output) ** 2)

    print('*')
    
    return error, 


### Starting point of the execution ###   

if __name__ == "__main__":

  ### Intializing the neuronal network ###

  Nperpop = 50  # Number of Neuron per population

  seed = 42               # Seed of the random number generator
  np.random.seed(seed)

  # Conducatonces of the different neuronal receptors

  gAMPA1 = 18.0
  gAMPA2 = 8.0
  gNMDA1 = 5.0
  gNMDA2 = 0.4
  gGABAA = 10.0
  gGABAB = 10.0

  # Stariting the scope. From this point onward, executed network architecture can be stored to restore.

  start_scope()

  # Dynamics of the membrane potential of the neurons

  eqs = '''
  dv/dt = (stimulus(t)*stimcoeff + g_leak*(e_leak-v) + s_ampatot*(0-v) + s_nmdatot*(0-v)/(1+exp(-0.062*v)/3.57) + s_gabaAtot*(-70-v) + s_gabaBtot*(-90-v))/tau : 1 (unless refractory)
  I : 1
  tau : second
  s_ampatot : 1
  s_nmdatot : 1
  s_gabaAtot : 1
  s_gabaBtot : 1
  stimcoeff : 1
  e_leak : 1
  g_leak : 1
  '''
  # Dynamics related to synpatic junctions

  eq_syn = '''
  ds_ampa/dt = -s_ampa/tau_ampa : 1
  ds_nmda/dt = -s_nmda/tau_nmda : 1
  ds_gabaA/dt = -s_gabaA/tau_gabaA : 1
  ds_gabaB/dt = -s_gabaB/tau_gabaB : 1
  s_ampatot_post =  g_ampa*s_ampa   : 1 (summed)
  s_nmdatot_post =  g_nmda*s_nmda   : 1 (summed)
  s_gabaAtot_post =  g_gabaA*s_gabaA   : 1 (summed)
  s_gabaBtot_post =  g_gabaB*s_gabaB   : 1 (summed)
  tau_ampa : second
  tau_nmda : second
  tau_gabaA : second
  tau_gabaB : second
  alpha_ampa : 1
  alpha_nmda : 1
  alpha_gabaA : 1
  alpha_gabaB : 1
  g_ampa : 1
  g_nmda : 1
  g_gabaA : 1
  g_gabaB : 1
  '''

  eq_syn_on_pre = '''
  s_ampa += alpha_ampa
  s_nmda += alpha_nmda
  s_gabaA += alpha_gabaA
  s_gabaB += alpha_gabaB
  '''

  # Setting up the neuronal groups
  
  G = NeuronGroup(4*Nperpop, eqs, threshold='v>-40', reset='v = -80', refractory=10*ms, method='rk4')
  G.v = -80
  G.tau = [10]*(4*Nperpop)*ms
  G.stimcoeff = [1]*Nperpop+[1.5]*Nperpop+[0]*Nperpop+[0]*Nperpop
  G.g_leak = 2.0
  G.e_leak = -80

  # Connecting synapses randomly

  S = Synapses(G, G, model = eq_syn, on_pre = eq_syn_on_pre)

  M = r_[c_[zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), np.random.rand(Nperpop,Nperpop) < 0.5, zeros([Nperpop,Nperpop])],
        c_[zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), np.random.rand(Nperpop,Nperpop) < 0.5],
        c_[zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop])], 
        c_[zeros([Nperpop,Nperpop]), zeros([Nperpop,Nperpop]), np.random.rand(Nperpop,Nperpop) < 0.5, zeros([Nperpop,Nperpop])]]

  taus = []
  g_ampas = []
  g_nmdas = []
  g_gabaAs = []
  g_gabaBs = []

  for iy in range(0,4*Nperpop):
    for ix in range(0,4*Nperpop):
        if M[iy,ix]:
            if iy < Nperpop and ix >= 2*Nperpop and ix < 3*Nperpop:       #From pop 0 (CRN) to pop 2 (PnC)
                S.connect(i=iy,j=ix)
                g_ampas.append(gAMPA1/Nperpop)
                g_nmdas.append(gNMDA1/Nperpop)
                g_gabaAs.append(0.0)
                g_gabaBs.append(0.0)
            elif iy < 2*Nperpop and ix >= 3*Nperpop:                      #From pop 1 (CN) to pop 3 (PPTg)
                S.connect(i=iy,j=ix)
                g_ampas.append(gAMPA2/Nperpop)
                g_nmdas.append(gNMDA2/Nperpop)
                g_gabaAs.append(0.0)
                g_gabaBs.append(0.0)
            elif iy >= 3*Nperpop and ix >= 2*Nperpop and ix < 3*Nperpop:  #From pop 3 (PPTg) to pop 2 (PnC)
                S.connect(i=iy,j=ix)
                g_ampas.append(0.0)
                g_nmdas.append(0.0)
                g_gabaAs.append(gGABAA/Nperpop)
                g_gabaBs.append(gGABAB/Nperpop)
            else:
                print("Somethings wrong")
                  
  S.tau_ampa = 5*msecond
  S.tau_nmda = 50*msecond
  S.tau_gabaA = 10*msecond
  S.tau_gabaB = 500*msecond
  S.alpha_ampa = 1.0
  S.alpha_nmda = 1.0
  S.alpha_gabaA = 1.0
  S.alpha_gabaB = 1.0

  ampa_ranges = find_sequence_ranges(g_ampas)
  nmda_ranges = find_sequence_ranges(g_nmdas)
  gabaa_ranges = find_sequence_ranges(g_gabaAs)
  gabab_ranges = find_sequence_ranges(g_gabaBs)


  mon = SpikeMonitor(G)         

  store()                 # Store the intialize neural network

  # Calculate the final simulated output using the best parameters
  def simulation(individual):

      gAMPA1, gAMPA2, gNMDA1, gNMDA2, gGABAA, gGABAB = individual
      
      for i in range(3):
        start_index,end_index = ampa_ranges[i]
        if i == 0:
          new_value = gAMPA1/Nperpop
        elif i == 1:
          new_value = gAMPA2/Nperpop
        elif i == 2:
          new_value = 0.0
        change_range_values(g_ampas, start_index, end_index, new_value)

      for i in range(3):
        start_index,end_index = nmda_ranges[i]
        if i == 0:
          new_value = gNMDA1/Nperpop
        elif i == 1:
          new_value = gNMDA2/Nperpop
        elif i == 2:
          new_value = 0.0
        change_range_values(g_nmdas, start_index, end_index, new_value)

      for i in range(2):
        start_index,end_index = gabaa_ranges[i]

        if i == 0:
          new_value = 0.0
        elif i == 1:
          new_value = gGABAA/Nperpop

        change_range_values(g_gabaAs, start_index, end_index, new_value)

      for i in range(2):
        start_index,end_index = gabab_ranges[i]

        if i == 0:
          new_value = 0.0
        elif i == 1:
          new_value = gGABAB/Nperpop

        change_range_values(g_gabaBs, start_index, end_index, new_value)

      startle_percent = []

      for isi in ISI:
          
          if isi == 0:
              sti = 0
          else:
              sti = 1

          restore()
          S.g_ampa = g_ampas
          S.g_nmda = g_nmdas
          S.g_gabaA = g_gabaAs
          S.g_gabaB = g_gabaBs
                      
          tstop = isi+120

          stimulus,_ = stimuli(sti,isi)          

          run(tstop*ms)
          
          PnC_counter = 0
          for i in array(mon.i):
              if i >= 100 and i < 150:
                  PnC_counter = PnC_counter + 1
          
          startle_percent.append(PnC_counter)
          
      if startle_percent[0] == 0.0:
          startle_percent[0] = 1e-6
          
      simulated_output = [(element / startle_percent[0])*100 for element in startle_percent[1:]]
      
      return simulated_output 


  ############################################################################# Genetic Algorithm Setup ############################################################################

  # Reference output data (replace this with your own reference output)
  reference_output = np.array([113,36,26,18,19,18,19.5,27,31,47.5,90,118])
  ISI = [0,4,8,12,20,30,50,100,300,500,1000,2000,5000]

  # Create a fitness minimizing class (since we want to minimize the error)
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("attr_float", np.random.uniform, 0, 30)  # Adjust bounds as needed
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", evaluate)
  toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Adjust alpha for blending
  toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Adjust sigma and indpb
  toolbox.register("select", tools.selTournament, tournsize=3)  # Adjust tournament size

  # Genetic Algorithm Execution
  population_size = 50
  num_generations = 1
  cxpb, mutpb = 0.7, 0.2


  initial_parameters = np.array([21.0191298, 9.3621201, 1.09513939, 1.6, 7.03380911, 5.41776811])     #Set the known best conductances to this array

  # Initialize the population with the manually defined parameters
  population = toolbox.population(n=population_size)
  for ind in population:
      ind[:] = initial_parameters

  # Run the optimization iteratively
  num_iterations = 2  # Set the number of iterations

  for iteration in range(num_iterations):
      population, logbook = algorithms.eaSimple(
          population, toolbox, cxpb, mutpb, num_generations, stats=None
      )
      
      # Get the best individual after the current iteration
      best_individual = tools.selBest(population, k=1)[0]
      best_fitness = best_individual.fitness.values[0]
      
      # Print the results for the current iteration
      print(f"Iteration {iteration + 1}: Best Individual {best_individual}, Fitness: {best_fitness}")

  # Best individual contains the optimized parameters from the last iteration
  final_best_individual = tools.selBest(population, k=1)[0]
  print("Final Best Parameters:", final_best_individual)


  final_simulated_output = simulation(final_best_individual)
  print("Final Simulated Output:", final_simulated_output)

  # Compare the simulated output with the reference output
  print("Reference Output:", reference_output)
  print("Difference:", final_simulated_output - reference_output)
