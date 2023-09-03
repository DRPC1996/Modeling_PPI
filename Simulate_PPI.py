#Import libraries

import matplotlib.pyplot as plt

from brian2 import *
from brian2tools import *

#### Generate stimulies related to PPI Experiments #####

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

#### Start Simulation ####

# sti => typeOfstimulus (0: Startling stimulus, 1: Startiling Stimulus followed by a pre pulse, 2: Only the pre-pulse)
# isi => inter-stimulus-interval in ms

def simulate(duration,sti,isi):
    
    restore()

    stimulus,P = stimuli(sti,isi)          

    run(duration*ms)

    return mon,P


#### Plot Results ####

# result => data recorded in spike monitor
# simulus => interpretable data of the relavant stimulus

def plot_results(result,stimulus):

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    for x, y in zip(result.t / msecond, array(result.i)):

        # Set different colors based on the y-value
        
        if y < 50:
            color = 'red'
            marker = 'D'
        elif y < 100:
            color = 'blue'
            marker = '+'
        elif y < 150:
            color = 'green'
            marker = 's'
        elif y < 200:
            color = 'orange'
            marker = 'H'

        # Plot the point
        axs[0].set_title('Neural Spikes Over Time')
        axs[0].plot(x, y, color=color, marker=marker, lw=0.5, ms=5, mew=0.5)
        axs[0].set_xlim([0, 200])
        axs[0].set_ylim([-5, 205])
        axs[0].legend(handles=[
            Line2D([], [], color='orange', marker='H', linestyle=''),
            Line2D([], [], color='g', marker='s', linestyle=''),
            Line2D([], [], color='b', marker='+', linestyle=''),
            Line2D([], [], color='r', marker='D', linestyle=''),
        ],
            labels=['PPTg', 'PnC', 'CN', 'CRN'], loc='upper right')

    axs[1].plot(stimulus)
    axs[1].set_xlim([0, 200])
    axs[1].set_ylim([0, 105])
    axs[1].set_xlabel('Time [ms]')

    plt.tight_layout()
    plt.show()

### Starting point of the execution ###   

if __name__ == "__main__":
    
    ### Intializing the neuronal network ###

    Nperpop = 50            # Number of Neuron per population

    seed = 42               # Seed of the random number generator
    np.random.seed(seed)

    # Conducatonces of the different neuronal receptors

    gAMPA1 = 21.0191298 
    gAMPA2 = 9.3621201 
    gNMDA1 = 1.09513939
    gNMDA2 = 1.6
    gGABAA = 7.03380911 
    gGABAB = 5.41776811

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

    g_ampas = []
    g_nmdas = []
    g_gabaAs = []
    g_gabaBs = []

    for iy in range(0,4*Nperpop):
        for ix in range(0,4*Nperpop):
            if M[iy,ix]:
                if iy < Nperpop and ix >= 2*Nperpop and ix < 3*Nperpop:         # From population 0 (CRN) to population 2 (PnC)
                    S.connect(i=iy,j=ix)
                    g_ampas.append(gAMPA1/Nperpop)
                    g_nmdas.append(gNMDA1/Nperpop)
                    g_gabaAs.append(0.0)
                    g_gabaBs.append(0.0)
                elif iy < 2*Nperpop and ix >= 3*Nperpop:                        # From population 1 (CN) to population 3 (PPTg)
                    S.connect(i=iy,j=ix)
                    g_ampas.append(gAMPA2/Nperpop)
                    g_nmdas.append(gNMDA2/Nperpop)
                    g_gabaAs.append(0.0)
                    g_gabaBs.append(0.0)
                elif iy >= 3*Nperpop and ix >= 2*Nperpop and ix < 3*Nperpop:    # From population 3 (PPTg) to population 2 (PnC)
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

    S.g_ampa = g_ampas
    S.g_nmda = g_nmdas
    S.g_gabaA = g_gabaAs
    S.g_gabaB = g_gabaBs

    S.alpha_ampa = 1.0
    S.alpha_nmda = 1.0
    S.alpha_gabaA = 1.0
    S.alpha_gabaB = 1.0

    mon = SpikeMonitor(G) 

    store()                                 # Store the intialize neural network

    result,stimulus = simulate(200,1,50)    # Calling the simulation 
    plot_results(result,stimulus)           # Plotting the Results