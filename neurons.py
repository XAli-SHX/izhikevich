# neurons.py

# Import libraries
import numpy as np  # numpy

# Izhikevich Neuron Object
class IzhikevichNeuron:
    def __init__(self, layer, weights, bias, steps, dt, neuron_type='regular_spiking'):
        self.layer = layer                          # Layer of Neuron
        self.weights = weights                      # Weights
        self.bias = bias                            # Bias

        # Izhikevich Neuron Parameters
        self.steps = steps                          # Number of steps for calculation
        self.dt = dt                                # Time step size
        self.v = np.zeros([self.steps])             # Membrane potential
        self.u = np.zeros([self.steps])             # Recovery variable
        self.spikes = np.zeros([self.steps])        # Output spike train
        self.n_spikes = 0                           # Number of spikes

        # Set initial conditions
        self.v[0] = -65.0                           # Initial membrane potential (mV)
        self.u[0] = 0.2 * self.v[0]                 # Initial recovery variable

        # Set neuron parameters based on type
        self.set_neuron_parameters(neuron_type)

    def set_neuron_parameters(self, neuron_type):
        if neuron_type == 'regular_spiking':
            self.a = 0.02
            self.b = 0.2
            self.c = -65
            self.d = 8
        elif neuron_type == 'fast_spiking':
            self.a = 0.1
            self.b = 0.2
            self.c = -65
            self.d = 2
        elif neuron_type == 'intrinsically_bursting':
            self.a = 0.02
            self.b = 0.2
            self.c = -55
            self.d = 4
        else:
            # Default to regular spiking
            self.a = 0.02
            self.b = 0.2
            self.c = -65
            self.d = 8

    def reset(self):
        # Reset neuron state for a new input
        self.v.fill(-65.0)
        self.u.fill(0.2 * self.v[0])
        self.spikes.fill(0)
        self.n_spikes = 0

    def calculate(self, neuron_input):
        for i in range(self.steps - 1):
            # Calculate total input current I
            I = self.bias
            for j in range(len(neuron_input)):
                if neuron_input[j][i]:
                    I += self.weights[j]

            # Update membrane potential and recovery variable using Euler method
            v = self.v[i]
            u = self.u[i]
            dv = (0.04 * v ** 2 + 5 * v + 140 - u + I) * self.dt
            du = (self.a * (self.b * v - u)) * self.dt

            self.v[i + 1] = v + dv
            self.u[i + 1] = u + du

            if self.v[i + 1] >= 30:
                self.v[i] = 30  # Spike
                self.v[i + 1] = self.c  # Reset membrane potential
                self.u[i + 1] += self.d  # Reset recovery variable
                self.spikes[i] = 1
                self.n_spikes += 1
            else:
                self.spikes[i] = 0



# Neuron Object for ANN
class Neuron:
    def __init__(self, layer, weights, bias):
        self.layer = layer                          # layer
        self.weights = weights                      # weights
        self.bias = bias                            # bias
        self.output = 0                             # output

    def calculate(self, inputs):                    # calculation
        relu_activation = ReLU()
        x = np.dot(self.weights, inputs) + self.bias
        self.output = relu_activation.calculate(x)


# ReLU activation function
class ReLU:
    def __init__(self):
        self.output = 0

    def calculate(self, x):
        self.output = np.maximum(0, x)              # relu function
        return self.output
