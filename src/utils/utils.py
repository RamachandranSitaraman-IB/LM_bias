from transformers import GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
activations = {}
import torch

def get_activation(name):
    def hook(model, input, output):
        if isinstance(output, list):
            output = output
        elif isinstance(output, torch.Tensor):
            output = output.detach()
        activations[name] = output #.detach()
    return hook, activations



def register_hook(model, layer):
    model_name = "gpt2"
    #model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Assuming we're interested in the last hidden layer
    layer_name = "transformer.h.["+str(layer)+"]"  # The last layer in GPT-2's transformer block
    hook, activations = get_activation((layer_name))
    model.transformer.h[layer].register_forward_hook(hook)
    return activations, layer_name


def heatmap(activations, layer_name, ratio, prompt_text):
    # Convert the activations to a format suitable for Seaborn's heatmap function
    # For GPT-2, activations are stored in a tensor with shape (batch_size, sequence_length, hidden_size)
    # We'll average across the batch dimension if there's more than one example
    print(activations.keys())
    if isinstance(activations[layer_name], torch.Tensor):
        # Detach and convert to numpy if it's a tensor
        activations[layer_name] = list(activations[layer_name].detach().cpu().numpy())
    elif isinstance(activations[layer_name], list):
        # Directly convert to numpy array if it's a list
        activations[layer_name] = activations[layer_name][0]

    # Now, activations[layer_name] should be a numpy array and can be manipulated as such
    activation_tensor = activations[layer_name].squeeze(0)  # Remove batch dim

    plt.figure(figsize=(10, 10))
    sns.heatmap(activation_tensor.cpu().detach().numpy(), cmap='viridis')
    plt.title(f"Heatmap of activations in layer {layer_name}")
    plt.xlabel("Neurons in the layer")
    plt.ylabel("Tokens in the sequence")
    #plt.show()
    plt.savefig("../../data/figs/"+ prompt_text+"-activation_heatmap"+layer_name+"-"+str(ratio)+".png")
