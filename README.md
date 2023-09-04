# Modeling_PPI
Welcome to the repository for the "Modeling Prepulse Inhibition (PPI)" project. This project focuses on providing an abstract simulation framework for studying and understanding the neural mechanisms underlying PPI using the Brian neural network simulator.

## Introduction

This project focuses on simulating a neural network using the Brian2 and Brian2tools libraries. The neural network model includes for neuronal populations (CRN, CN, PPTg and PnC) and four neuronal receptors (AMPA,NMDA,GABAA,GABAB). Codes in this repositorty simulate the dynamic of the network in response to specific stimuli. There are three python files as mentioned in File Description.

## Requirements

To run this codes, you'll need:

- Python (>=3.6)
- Brian2 library
- Brian2tools library
- DEAP (Distributed Evolutionary Algorithms in Python) library
- NumPy library
- Matplotlib library

You can install the required libraries using the following command:

pip install brian2 brian2tools deap numpy matplotlib

## File Descriptions

- README.md: The README file you're currently reading.
- Simulate_PPI.py => This will generate a figure of spkies of each related neuron populations (CRN, CN, PPTg and PnC) with the specific stimuli.
- ISI_Curve.py => Interstimulus delay between prepulse and startle pulse is significant for PPI mechanism. This code will generate a plot of starle percent (activation percentage of PnC population) against interstimulus delay.
- Optimize_Conductances.py => Here a genetic algorithm is employed to optimize the conductance parameters of the neural network simulation. The goal of the optimization process is to find the optimal conductance values for different neuronal receptors that result in a simulated output closely matching a reference output.

## Related Works

Along the project me and my research partner work together to implement two models. I was modeling the PPI phenomenon and my partner was modeling the Mis-Match Negativity otherwise MMN phenomenon. Following, link will direct to a repository which includes that work. 

{Link will be updated}

## Acknowledgements

We extend our heartfelt gratitude to the Computational Neuroscience Group for their unwavering support, invaluable insights, and the unparalleled expertise they provided throughout the duration of this study. Special thanks are due to our dedicated supervisors, Professor Tuomo Mäki-Marttunen and Professor Marja-Leena Linne. Their profound knowledge, meticulous guidance, and unyielding commitment were instrumental in navigating the complexities of this research. The enriching environment they fostered, coupled with their mentorship, significantly shaped this work and inspired us at every juncture. 

This project works were funded by Academy of Finland, grant numbers 330776 and 336376.

