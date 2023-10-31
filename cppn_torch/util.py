import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
import inspect
import random
from torch import nn
import torch
from typing import List, Union
from cv2 import resize as cv2_resize

from cppn_torch.cppn import NodeType
from cppn_torch.graph_util import feed_forward_layers, get_ids_from_individual, get_incoming_connections_weights, required_for_output
from cppn_torch.normalization import handle_normalization
   
from torchvision.transforms import GaussianBlur
   
def visualize_network(individual, config, sample_point=None, color_mode="L", visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None, curved=False, return_fig=False):
    c = config
    if(sample):
        if sample_point is None:
            sample_point = [.25]*c.num_inputs
        individual.eval(sample_point)
            
        
    nodes = individual.node_genome
    connections = individual.connection_genome.items()

    if not visualize_disabled:
        req = required_for_output(*get_ids_from_individual(individual))
        nodes = {k: v for k, v in nodes.items() if v.id in req or v.type == NodeType.INPUT or v.type == NodeType.OUTPUT}

    max_weight = c.max_weight

    G = nx.DiGraph()
    function_colors = {}
    colors = ['lightsteelblue'] * len([node.activation for node in individual.node_genome.values()])
    node_labels = {}

    node_size = 2000
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    for i, fn in enumerate([node.activation for node in individual.node_genome.values()]):
        if not hasattr(fn, '__name__'):
            fn.__name__ =  str(type(fn))
        function_colors[fn.__name__] = colors[i]
        
    function_colors["identity"] = colors[0]

    fixed_positions={}
    inputs = {k:v for k,v in nodes.items() if v.type==NodeType.INPUT}
    for i, node in enumerate(inputs.values()):
        if node.type == NodeType.INPUT:
            if not visualize_disabled and node.layer == 999:
                continue
            G.add_node(node, color=function_colors[node.activation.__name__], shape='d', layer=(node.layer))
            if node.type == 0:
                node_labels[node] = f"input{i}\n{node.id}"
                
            fixed_positions[node] = (-4,((i+1)*2.)/len(inputs))

    for node in nodes.values():
        if node.type == NodeType.HIDDEN:
            if not visualize_disabled and node.layer == 999:
                continue
            G.add_node(node, color=function_colors[node.activation.__name__], shape='o', layer=(node.layer))
            node_labels[node] = f"{node.id}\n{node.activation.__name__}"

    for i, node in enumerate(nodes.values()):
        if node.type == NodeType.OUTPUT:
            # if not visualize_disabled and node.layer == 999:
                # continue
            title = i
            G.add_node(node, color=function_colors[node.activation.__name__], shape='s', layer=(node.layer))
            node_labels[node] = f"{node.id}\noutput{title}:\n{node.activation.__name__}"
            fixed_positions[node] = (4, ((i+1)*2)/len(individual.output_nodes()))
    pos = {}
    fixed_nodes = fixed_positions.keys()
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4,subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    plt.figure(figsize=(10, 10))

    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        this_nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in this_nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=this_nodes)

    edge_labels = {}
    for _, cx in connections:
        key = cx.key_tuple
        if key[0] not in nodes.keys() or key[1] not in nodes.keys():
            continue
        w = cx.weight.item()
        if not visualize_disabled and not cx.enabled:
        # or np.isclose(w, 0))): 
            continue
        style = ('-', 'k',  .5+abs(w)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(w)/max_weight)
        if(cx.enabled and w<0): style  = ('-', 'r', .5+abs(w)/max_weight)
        from_node = nodes[key[0]]
        to_node = nodes[key[1]]
        if from_node in G.nodes and to_node in G.nodes:
            G.add_edge(from_node, to_node, weight=f"{w:.4f}", pos=pos, style=style)
        else:
            print("Connection not in graph:", from_node.id, "->", to_node.id)
        edge_labels[(from_node, to_node)] = f"{w:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))
    use_curved = curved
    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                connectionstyle= "arc3" if not use_curved else f"arc3,rad={0.2*random.random()}",
                                # connectionstyle= "arc3"
                            )
    
    if extra_text is not None:
        plt.text(0.5,0.05, extra_text, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        
    
    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    if return_fig:
        return plt.gcf()
    elif save_name is not None:
        plt.savefig(save_name, format="PNG")
    else:
        plt.show()


def print_net(individual, show_weights=False, visualize_disabled=False):
    print(f"<CPPN {individual.id}")
    print(f"nodes:")
    for k, v in individual.node_genome.items():
        print("\t",k, "\t|\t",v.layer, "\t|\t",v.activation.__name__)
    print(f"connections:")
    for k, v in individual.connection_genome.items():
        print("\t",k, "\t|\t",v.enabled, "\t|\t",v.weight)
    print(">")
  
  
def get_network_images(networks):
    imgs = []
    for net in networks:
        fig = visualize_network(net, return_fig=True)
        ax = fig.gca()
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(img)  
    return imgs


def get_max_number_of_hidden_nodes(population):
    max = 0
    for g in population:
        if len(list(g.hidden_nodes()))> max:
            max = len(list(g.hidden_nodes()))
    return max

def get_avg_number_of_hidden_nodes(population):
    count = 0
    for g in population:
        count+=len(g.node_genome) - g.n_in_nodes - g.n_outputs
    return count/len(population)

def get_max_number_of_connections(population):
    max_count = 0
    for g in population:
        count = len(list(g.enabled_connections()))
        if(count > max_count):
            max_count = count
    return max_count

def get_min_number_of_connections(population):
    min_count = math.inf
    for g in population:
        count = len(list(g.enabled_connections())) 
        if(count < min_count):
            min_count = count
    return min_count

def get_avg_number_of_connections(population):
    count = 0
    for g in population:
        count+=len(list(g.enabled_connections()))
    return count/len(population)



def upscale_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, device="cpu"):
    # return ConvTranspose2d(in_channels,out_channels,kernel_size, stride=stride, padding=padding, output_padding=1,device=device)
    layer = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, device=device, bias=bias)
   )
    return layer

def show_inputs(inputs, cols=8, cmap='viridis'):
    if not isinstance(inputs, torch.Tensor):
        # assume it's an algorithm instance
        inputs = inputs.inputs
        try:
            inputs = handle_normalization(inputs, inputs.config.normalize_outputs)
        except:
            pass # no normalization
    inputs = inputs.permute(2,0,1)
    image_grid(inputs,
               cols=cols,
               show=True,
               cmap=cmap,
               suptitle="Inputs")


def image_grid(images,
                cols=4,
                titles=None,
                show=True,
                cmap='gray',
                suptitle=None,
                title_font_size=12,
                fig_size=(10,10)):
    
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        images = [i for i in images]
    fg = plt.figure(constrained_layout=True, figsize=fig_size)
    rows = 1 + len(images) // cols
    for i, img in enumerate(images):
        ax = fg.add_subplot(rows, cols, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if titles is not None:
            ax.set_title(titles[i])
    if suptitle is not None:
        fg.suptitle(suptitle, fontsize=title_font_size)
    if show:
        fg.show()
    else:
        return plt.gcf()


def custom_image_grid(images:Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               cols=8, titles=None, show=True, cmap="gray"):
    assert titles is None or len(titles) == len(images)
    if isinstance(images, List):
        images = torch.stack(images).detach().cpu()
    elif isinstance(images, np.ndarray):
        ...
    elif isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    
    num = images.shape[0]
    
    rows = math.ceil(num / cols)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*2, rows*2))
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
        if i >= num:
            ax.imshow(np.ones((num, images.shape[1], 3)), cmap="gray")
        else:    
            ax.imshow(images[:, :, i], cmap=cmap, vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(f"Input {titles[i]}")
    if show:
        fig.tight_layout()
        fig.show()
    return fig
        
 



def visualize_node_outputs(net, inputs):
    batch_size = 1 
    
    if inputs is None:
        inputs = type(net).constant_inputs

    res_h = inputs.shape[0]
    res_w = inputs.shape[1]
    
    # reset the activations to 0 before evaluating
    net.reset_activations()
    
    node_outputs = {}
 
    # get layers
    layers = feed_forward_layers(net) 
    layers.insert(0, net.input_nodes().keys()) # add input nodes as first layer
    
    for i, p_layer in enumerate(net.pre_layers):
        inputs = inputs.permute(2,0,1)
        inputs = p_layer(inputs)
        inputs = torch.relu(inputs)
        inputs = inputs.permute(1, 2, 0)
        
        node_outputs[net.layer_to_str(p_layer, i)] = inputs.clone().detach().cpu().squeeze(0)
    
    # iterate over layers
    for layer in layers:
        Xs, Ws, nodes, Fns = [], [], [], []
        for node_index, node_id in enumerate(layer):
            # iterate over nodes in layer
            node = net.node_genome[node_id] # the current node
            
            if node.type == NodeType.INPUT:
                # initialize the node's sum
                X = inputs[:,:,node_index].repeat(batch_size, 1, 1, 1) # (batch_size, cx, res_h, res_w)
                weights = torch.ones((1), dtype=net.config.dtype, device=net.device)
            else:
                # find incoming connections and activate
                X, weights = get_incoming_connections_weights(net, node)
                # X shape = (batch_size, num_incoming, res_h, res_w)
                if X is None:
                    X = torch.zeros((batch_size, 1, res_h, res_w), dtype=net.config.dtype, device=net.device)
                if weights is None:
                    weights = torch.ones((1), dtype=net.config.dtype, device=net.device)

            node.activate(X, weights) # naive
            print("activate", node.id, node.outputs.shape)
            node_outputs[node.id] = node.outputs.clone().detach().cpu().squeeze(0)
            
    # collect outputs from the last layer
    sorted_o = sorted(net.output_nodes().values(), key=lambda x: x.key, reverse=True)
    outputs = torch.stack([node.outputs for node in sorted_o], dim=1)
    assert str(outputs.device) == str(net.device), f"Output is on {outputs.device}, should be {net.device}"

    for i, layer in enumerate(net.post_layers):
        if isinstance(layer, torch.nn.Linear):
            outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = layer(outputs)
        outputs = torch.relu(outputs)
        if isinstance(layer, torch.nn.Linear):
            outputs = outputs.reshape(outputs.shape[0], len(net.config.color_mode), net.config.res_h, net.config.res_w)
        node_outputs[net.layer_to_str(layer, i)] = outputs.clone().detach().cpu().squeeze(0).permute(1, 2, 0)
            
    net.outputs = outputs
    # net._pre_post_outputs = net.outputs.clone()
    
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    fig = plt.figure(figsize=(30,30))
    G = net.to_nx()
    pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
    ax=plt.gca()
    fig=plt.gcf()
    imsize = 0.03 # this is the image size
    nx.draw_networkx(
        G,
        with_labels=True,
        pos=pos,
        labels={n:f"{n}\n{net.node_genome[n].activation.__name__[:4]}\n{1}xHxW"if n in net.node_genome else n
                    for n in G.nodes(data=False) },
        node_size=6000,
        font_size=6,
        node_shape='s',
        node_color=['lightsteelblue' if n in net.node_genome else 'lightgreen' for n in G.nodes()  ]
        )
    plt.annotate('# params: ' + str(net.num_params), xy=(1.0, 1.0), xycoords='axes fraction', fontsize=12, ha='right', va='top')
    # plt.show()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    for n in G.nodes():
        if not n in node_outputs.keys():
            print(f"Node {n} not in node_outputs")
            continue
        (x,y) = pos[n]
        xx,yy = trans((x,y)) # figure coordinates
        xa,ya = trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
        
        color = 'gray'
        if net.config.color_mode == 'RGB':
            S = sorted(net.output_nodes().keys(), reverse=True)
            if n in S:
                idx = S.index(n)
                if idx == 0:
                    color =  colors.LinearSegmentedColormap.from_list('mycmap', ['black', 'red'])
                elif idx == 1:
                    color =  colors.LinearSegmentedColormap.from_list('mycmap', ['black', 'green'])
                elif idx == 2:
                    color =  colors.LinearSegmentedColormap.from_list('mycmap', ['black', 'blue'])

        img = node_outputs[n]
        if len(img.shape)<=2 or img.shape[-1] <= 3:
            a.imshow(img, cmap=color)
            a.set_aspect('equal')
            a.axis('off')
        else:
            # multiple channels:
            print('multiple channels')
            print(img.shape)
            num_channels = img.shape[-1]
            size = imsize/np.sqrt(num_channels)
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = int(np.ceil(num_channels / rows))
            # start at top left and move right/down
            
            for i in range(num_channels):
                r = i // cols
                c = i % cols
                a = plt.axes([(xa+c*size)- imsize/2.0,(ya+r*size)- imsize/2.0, size, size ])
                a.set_aspect('equal')
                a.axis('off')
                a.imshow(img[:,:,i], cmap=color)
                a.set_aspect('equal')
                a.axis('off')
    plt.show()
        

def gaussian_blur(img, sigma, kernel_size=(5,5)):
    return GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        
        
def resize(img, size):
    return cv2_resize(img, size)

def center_crop(img, r, c):
    h, w = img.shape[:2]
    r1 = int(round((h - r) / 2.))
    c1 = int(round((w - c) / 2.))
    return img[r1:r1 + r, c1:c1 + c]

def random_uniform(generator, low=0.0, high=1.0, grad=False):
    if generator:
        return ((low - high) * torch.rand(1, device=generator.device, requires_grad=grad, generator=generator) + high)[0]
    else:
        return ((low - high) * torch.rand(1, requires_grad=grad) + high)[0]
    
def random_normal (generator=None, mean=0.0, std=1.0, grad=False):
    if generator:
        return torch.randn(1, device=generator.device, requires_grad=grad, generator=generator)[0] * std + mean
    else:
        return torch.randn(1, requires_grad=grad)[0] * std + mean

# def random_choice(generator, choices, count, replace):
#     if not replace:
#         # if generator:
#             # indxs = torch.randperm(len(choices), generator=generator)[:count]
#         # else:
#         indxs = torch.randperm(len(choices))[:count]
#         output = []
#         for i in indxs:
#             output.append(choices[i])
#         return output
#     else:
#         if generator:
#             return choices[torch.randint(len(choices), (count,), generator=generator)]
#         else:
#             return choices[torch.randint(len(choices), (count,))]
        
def random_choice(options, count=1, replace=False):
    """Chooses a random option from a list of options"""
    if not replace:
        indxs = torch.randperm(len(options))[:count]
        output = []
        for i in indxs:
            output.append(options[i])
        if count == 1:
            return output[0]
        return output
    else:
        return options[torch.randint(len(options), (count,))]
    

def genetic_difference(cppn, other) -> float:
    # only enabled connections, sorted by innovation id
    this_cxs = sorted(cppn.enabled_connections(),
                        key=lambda c: c.key)
    other_cxs = sorted(other.enabled_connections(),
                        key=lambda c: c.key)

    N = max(len(this_cxs), len(other_cxs))
    other_innovation = [c.key for c in other_cxs]

    # number of excess connections
    n_excess = len(get_excess_connections(this_cxs, other_innovation))
    # number of disjoint connections
    n_disjoint = len(get_disjoint_connections(this_cxs, other_innovation))

    # matching connections
    this_matching, other_matching = get_matching_connections(
        this_cxs, other_cxs)
    
    difference_of_matching_weights = [
        abs(o_cx.weight.item()-t_cx.weight.item()) for o_cx, t_cx in zip(other_matching, this_matching)]
    # difference_of_matching_weights = torch.stack(difference_of_matching_weights)
    
    if(len(difference_of_matching_weights) == 0):
        difference_of_matching_weights = 0
    else:
        difference_of_matching_weights = sum(difference_of_matching_weights) / len(difference_of_matching_weights)

    # Furthermore, the compatibility distance function
    # includes an additional argument that counts how many
    # activation functions differ between the two individuals
    n_different_fns = 0
    for t_node, o_node in zip(cppn.node_genome.values(), other.node_genome.values()):
        if(t_node.activation.__name__ != o_node.activation.__name__):
            n_different_fns += 1

    # can normalize by size of network (from Ken's paper)
    if(N > 0):
        n_excess /= N
        n_disjoint /= N

    # weight (values from Ken)
    n_excess *= 1
    n_disjoint *= 1
    difference_of_matching_weights *= .4
    n_different_fns *= 1
    
    difference = sum([n_excess,
                        n_disjoint,
                        difference_of_matching_weights,
                        n_different_fns])
    if torch.isnan(torch.tensor(difference)):
        difference = 0

    return difference
