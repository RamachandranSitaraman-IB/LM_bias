from transformers import GPT2Model, GPT2Tokenizer, OpenAIGPTTokenizer
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



def register_hook(model, layer, model_name):
    if model_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    else:
    #model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Assuming we're interested in the last hidden layer
    layer_name = "transformer.h.["+str(layer)+"]"  # The last layer in GPT-2's transformer block
    hook, activations = get_activation((layer_name))
    model.transformer.h[layer].register_forward_hook(hook)
    return activations, layer_name


def heatmap(activations, layer_name, ratio, prompt_text, model_name):
    if model_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    else:
    #model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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
    print(activation_tensor.shape)
    plt.figure(figsize=(10, 10))
    sns.heatmap(activation_tensor.cpu().detach().numpy(), cmap='viridis')
    plt.title(f"Heatmap of activations in layer {layer_name}")
    plt.xlabel("Neurons in the layer")
    plt.ylabel("Tokens in the sequence")
    #plt.show()
    plt.savefig("../../data/figs/"+ prompt_text+"-activation_heatmap"+layer_name+"-"+str(ratio)+".png")
    plt.close()

def heatmaptext(activations, layer_name, ratio, prompt_text, input_ids, model_name):

    if model_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    else:
    #model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model_name = "gpt2"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize the prompt text to get the token texts for y-axis labels
    #token_texts = [tokenizer.decode([token_id], clean_up_tokenization_spaces=True) for token_id in input_ids.tolist()[0] ]
    #token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]

    input_ids_sequence = input_ids.tolist()[0]  # Assuming input_ids is a tensor and ii indexes a specific sequence
    token_texts = tokenizer.convert_ids_to_tokens(input_ids_sequence)

    if isinstance(activations[layer_name], torch.Tensor):
        # Detach and convert to numpy if it's a tensor
        activations[layer_name] = list(activations[layer_name].detach().cpu().numpy())
    elif isinstance(activations[layer_name], list):
        # Directly convert to numpy array if it's a list
        activations[layer_name] = activations[layer_name][0]
    # Assuming activations[layer_name] is your tensor with shape [sequence_length, hidden_size]
    activation_tensor = activations[layer_name].squeeze(0)  # Assuming batch_size=1 for simplicity
    print(activations.keys(), len(token_texts), len(input_ids.tolist()[0]), activations[layer_name].shape)

    # Prepare the figure and plot the heatmap
    plt.figure(figsize=(10, 10))  # Adjust figure size as needed
    ax = sns.heatmap(activation_tensor.cpu().detach().numpy(), cmap='viridis', yticklabels=token_texts)

    plt.title(f"Heatmap of activations in layer {layer_name}")
    plt.xlabel("Neurons in the layer")
    plt.ylabel("Tokens in the sequence")

    # Rotate y-axis labels for readability
    plt.yticks(rotation=45, ha='right')  # ha is the horizontal alignment

    # Save or show the plot
    plt.savefig("../../data/figs/" + prompt_text + "-activation_heatmap" + layer_name + "-" + str(ratio) + ".png")
    # plt.show()
    plt.close()