#
#   Author: Klaus Niederberger
#   Release: Marco Winzker, Hochschule Bonn-Rhein-Sieg, 22.12.2022
#
# this software creates SNN parameters based on an existing ANN with ReLU activation

# import submodules
import normalization as norm
import functions as fct
import ann_model as model
import coding as code

# main.py

# ... existing imports

# Settings
steps = 1000                      # Number of time steps for simulation
dt = 1.0                          # Time step size in ms
img_name = 'test.png'             # Name of the image file
fp_factor = 2 ** 8                # Fixed-point scaling factor

print("Start Program")

# Read input image
original_img, img = fct.read_img(img_name)

# Process input
input_array, width, height = fct.process_img(img)

# Initialize network based on ANN
network_structure, ann_weights, ann_biases, class_th = model.load_model()

# Create ANN network
ann_neuron_array = fct.create_neurons(network_structure, ann_weights, ann_biases, 0, 0, 'ANN')

# Calculate ANN
print("Calculating ANN...")
output_array, max_activation = fct.calculate_network(network_structure, ann_neuron_array, input_array, 0, 'ANN')

# Create output image of ANN
output_img = fct.create_output_img(output_array, width, class_th)

# Data-based normalization (you might need to adjust this step)
weight_factor, bias_factor, v_th = norm.data_normalization(max_activation)

# Scale weights and biases according to normalization procedure
snn_weights, snn_biases = fct.scale_model(network_structure, ann_weights, ann_biases, weight_factor, bias_factor, 0)

# Create SNN network with Izhikevich neurons
neuron_type = 'regular_spiking'  # You can choose different neuron types
snn_neuron_array = fct.create_neurons(network_structure, snn_weights, snn_biases, steps, dt, 'SNN', neuron_type)

# Calculate SNN
print("Calculating SNN...")
output_spikes, not_used = fct.calculate_network(network_structure, snn_neuron_array, input_array, steps, 'SNN')

# Decode spikes of output neurons
spike_ratio = code.decode_output_spikes(output_spikes, steps)

# Create output image of SNN
sp_output_img = fct.create_output_img(spike_ratio, width, class_th / max_activation[-1])

# Write to file
fct.write_to_file(snn_weights, snn_biases, v_th, class_th / max_activation[-1], fp_factor, steps, network_structure)

# Show images
fct.show_output(original_img, output_img, sp_output_img)
