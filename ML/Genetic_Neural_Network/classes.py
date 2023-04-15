# main algorithm
import random as rand
import numpy as np
import networkx as nx  # generating graphs
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt  # visualizing graphs
import matplotlib.ticker as plticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PIL  # image processing
from hyperparams import *


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_np(x: float) -> float:
    return 2*sigmoid(x)-1

def sigmoid_prime(x: float, m: float) -> float:
    return sigmoid(x-m)


def null(__) -> float:
    return 0.0

def llun(__) -> float:
    return 1.0


def x(entity) -> float:
    return entity.x/CANVAS_DIM

def y(entity) -> float:
    return entity.y/CANVAS_DIM


def dist_n(entity) -> float:
    x, y = entity.x, entity.y
    nearest_y = max([wall[1][1] for wall in WALLS if ((wall[0][0] <= x <= wall[0][1]) and (wall[1][1] <= y))] + [0])
    return sigmoid_prime(y - nearest_y, CANVAS_DIM/8)

def dist_s(entity) -> float:
    x, y = entity.x, entity.y
    nearest_y = min([wall[1][0] for wall in WALLS if ((wall[0][0] <= x <= wall[0][1]) and (wall[1][0] >= y))] + [CANVAS_DIM])
    return sigmoid_prime(nearest_y - y, CANVAS_DIM/8)

def dist_e(entity) -> float:
    x, y = entity.x, entity.y
    nearest_x = min([wall[0][0] for wall in WALLS if ((wall[1][0] <= y <= wall[1][1]) and (wall[0][0] >= x))] + [CANVAS_DIM])
    return sigmoid_prime(nearest_x - x, CANVAS_DIM/8)

def dist_w(entity) -> float:
    x, y = entity.x, entity.y
    nearest_x = max([wall[0][1] for wall in WALLS if ((wall[1][0] <= y <= wall[1][1]) and (wall[0][1] <= x))] + [0])
    return sigmoid_prime(x - nearest_x, CANVAS_DIM/8)

_dist_ne = lambda entity: min(dist_n(entity), dist_e(entity))
_dist_nw= lambda entity: min(dist_n(entity), dist_w(entity))
_dist_se = lambda entity: min(dist_s(entity), dist_e(entity))
_dist_sw = lambda entity: min(dist_s(entity), dist_w(entity))

def dist_fwd(entity) -> float:
    dist_list = [dist_n, _dist_ne, dist_e, _dist_se, dist_s, _dist_sw, dist_w, _dist_nw]
    return dist_list[entity.direction](entity)


def nearest_d(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if (e != entity)] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def nearest_dx(entity) -> float:
    x = entity.x
    nearest_dist = min([abs(e.x - x) for e in entities if ((e != entity) and (e.x == x))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def nearest_dy(entity) -> float:
    y = entity.y
    nearest_dist = min([abs(e.y - y) for e in entities if ((e != entity) and (e.y == y))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_n(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and ((y - e.y) >= abs(e.x - x)))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_s(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and ((e.y - y) >= abs(e.x - x)))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_e(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and ((e.x - x) >= abs(e.y - y)))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_w(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and ((x - e.x) >= abs(e.y - y)))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_ne(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and (e.x >= x) and (e.y <= y))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_nw(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and (e.x <= x) and (e.y <= y))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_se(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and (e.x >= x) and (e.y >= y))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def _nearest_sw(entity) -> float:
    x, y = entity.x, entity.y
    nearest_dist = min([(abs(e.x - x) + abs(e.y - y)) for e in entities if ((e != entity) and (e.x <= x) and (e.y >= y))] + [float('inf')])
    return sigmoid_prime(nearest_dist, CANVAS_DIM/8)

def nearest_fwd(entity) -> float:
    nearest_list = [_nearest_n, _nearest_ne, _nearest_e, _nearest_se, _nearest_s, _nearest_sw, _nearest_w, _nearest_nw]
    return nearest_list[entity.direction](entity)


def elapsed_t(entity) -> float:
    return (frameCount / FRAMES_PER_GEN)

def population(entity) -> float:
    return (entities.alive / INITIAL_GEN_POP)

def oscillator(entity) -> float:
    return ((np.sin(2 * np.pi * entity.freq * frameCount) + 1.0) / 2)



def move_n(neuron):
    if neuron.value >= 0.6:
        neuron.master.entity.move_y(-neuron.value * K_VELOCITY)

def move_s(neuron):
    if neuron.value >= 0.6:
        neuron.master.entity.move_y(neuron.value * K_VELOCITY)

def move_e(neuron):
    if neuron.value >= 0.6:
        neuron.master.entity.move_x(neuron.value * K_VELOCITY)

def move_w(neuron):
    if neuron.value >= 0.6:
        neuron.master.entity.move_x(-neuron.value * K_VELOCITY)

def move_hrz(neuron):
    if neuron.value <= 0.25:
        neuron.master.entity.move_x(-(1.0 - 2*neuron.value) * K_VELOCITY)
    elif neuron.value >= 0.75:
        neuron.master.entity.move_x((2*neuron.value - 1.0) * K_VELOCITY)

def move_vrt(neuron):
    if neuron.value <= 0.25:
        neuron.master.entity.move_y(-(1.0 - 2*neuron.value) * K_VELOCITY)
    elif neuron.value >= 0.75:
        neuron.master.entity.move_y((2*neuron.value - 1.0) * K_VELOCITY)

_move_ne = lambda neuron: (move_n(neuron), move_e(neuron))[0]
_move_nw = lambda neuron: (move_n(neuron), move_w(neuron))[0]
_move_se = lambda neuron: (move_s(neuron), move_e(neuron))[0]
_move_sw = lambda neuron: (move_s(neuron), move_w(neuron))[0]

def move_fwd(neuron):
    if neuron.value >= 0.6:
        move_list = [move_n, _move_ne, move_e, _move_se, move_s, _move_sw, move_w, _move_nw]
        move_list[neuron.master.entity.direction](neuron)

def rotate(neuron):
    if 0.3 < neuron.value < 0.7:
        return
    if neuron.value <= 0.075:
        neuron.master.entity.direction = (neuron.master.entity.direction  - 2) % 8
        return
    if neuron.value <= 0.3:
        neuron.master.entity.direction = (neuron.master.entity.direction  - 1) % 8
        return
    if neuron.value >= 0.925:
        neuron.master.entity.direction = (neuron.master.entity.direction  + 2) % 8
        return
    if neuron.value >= 0.7:
        neuron.master.entity.direction = (neuron.master.entity.direction  + 1) % 8
        return
    
def set_freq(neuron):
    if neuron.value < 0.4:
        return
    neuron.master.entity.freq *= np.float_power(1.5,  (neuron.value - 0.7)/0.3)


class Neuron:
    def __init__(self, master, identifier: tuple[int, int], bias: float, name=""):  # master: 종속된 Entity
        self.master = master
        self.id = identifier
        self.value = bias
        self.bias = bias
        self.name = name

    def copy(self, master):
        return Neuron(master, self.id, self.bias, self.name)

    def reset(self):
        self.value = self.bias


class InputNeuron(Neuron):
    def __init__(self, master, identifier: tuple[int, int], bias: float, inputf=null, name=""):
        Neuron.__init__(self, master, identifier, bias, name)
        self.input = inputf

    def get_input(self):
        self.value = self.input(self.master.entity)

    def copy(self, master):
        return InputNeuron(master, self.id, self.bias, self.input, self.name)


class OutputNeuron(Neuron):
    def __init__(self, master, identifier: tuple[int, int], bias: float, function=None, name=""):
        Neuron.__init__(self, master, identifier, bias, name)
        self.function = function

    def copy(self, master):
        return OutputNeuron(master, self.id, self.bias, self.function, self.name)

    def act(self):
        self.function(self)
        self.reset()


InputNeurons = [
    InputNeuron(None, (0,0), 0.0, null, "0"),
    InputNeuron(None, (0,1), 0.0, llun, "1"),
    InputNeuron(None, (0,2), 0.0, x, "x"),
    InputNeuron(None, (0,3), 0.0, y, "y"),
    InputNeuron(None, (0,4), 0.0, dist_n, "dist_n"),
    InputNeuron(None, (0,5), 0.0, dist_s, "dist_s"),
    InputNeuron(None, (0,6), 0.0, dist_e, "dist_e"),
    InputNeuron(None, (0,7), 0.0, dist_w, "dist_w"),
    InputNeuron(None, (0,8), 0.0, dist_fwd, "dist_fwd"),
    InputNeuron(None, (0,9), 0.0, nearest_d, "nearest_d"),
    InputNeuron(None, (0,10), 0.0, nearest_dx, "nearest_dx"),
    InputNeuron(None, (0,11), 0.0, nearest_dy, "nearest_dy"),
    InputNeuron(None, (0,12), 0.0, nearest_fwd, "nearest_fwd"),
    InputNeuron(None, (0,13), 0.0, elapsed_t, "elapsed_t"),
    InputNeuron(None, (0,14), 0.0, population, "population"),
    InputNeuron(None, (0,15), 0.0, oscillator, "oscillator")            
]  # 미리 지정한 InputNeuron 들의 list

OutputNeurons = [
    OutputNeuron(None, (BRAIN_DEPTH + 1,0), 0.0, null, "null"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,1), 0.0, move_n, "move_n"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,2), 0.0, move_s, "move_s"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,3), 0.0, move_e, "move_e"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,4), 0.0, move_w, "move_w"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,5), 0.0, move_hrz, "move_hrz"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,6), 0.0, move_vrt, "move_vrt"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,7), 0.0, move_fwd, "move_fwd"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,8), 0.0, rotate, "rotate"), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,9), 0.0, set_freq, "set_freq"),
]
'''
    OutputNeuron(None, (BRAIN_DEPTH + 1,9), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,10), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,11), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,12), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,13), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,14), 0.0, null), 
    OutputNeuron(None, (BRAIN_DEPTH + 1,15), 0.0, null), '''
#] # 미리 지정한 OutputNeuron 들의 list

n_InputN = len(InputNeurons)
n_OutputN = len(OutputNeurons)


def interp_gene(network): 
    # set up the network using the gene
    l_c = [[] for i in range(BRAIN_DEPTH + 2)]
    d_ip = {}
    d_op = {}
    for n in range(0, BRAIN_SIZE):
        g = network.gene[6 * n: 6 * n + 6]
        g_t = int(g[0], 16) % (BRAIN_DEPTH + 3)

        if g_t > BRAIN_DEPTH + 1:
            continue
        g_so = int(g[1], 16)
        g_si = int(g[2], 16)
        g_w = MAX_WEIGHT * (float(int(g[3:-1], 16)) - 127.5) / 127.5

        if g_t == 0:
            # input -> hidden[0]
            g_so = g_so % n_InputN
            g_si = g_si % N_PER_LAYER
            d_ip[g_so] = 0
        elif g_t == BRAIN_DEPTH:
            # hidden[-1] -> output
            g_so = g_so % N_PER_LAYER
            g_si = g_si % n_OutputN
            d_op[g_si] = 0
        elif g_t == BRAIN_DEPTH + 1:
            # hidden[0] -> [1]
            g_t = 1
            g_so = g_so % N_PER_LAYER
            g_si = g_si % N_PER_LAYER
            '''
            # input -> output
            g_so = g_so % n_InputN
            g_si = g_si % n_OutputN
            d_ip[g_so] = 0
            d_op[g_si] = 0
            '''
        else:
            # hidden[i-1] -> hidden[i]
            g_so = g_so % N_PER_LAYER
            g_si = g_si % N_PER_LAYER
        l_c[g_t].append((g_so, g_si, g_w))

    for ip in d_ip:
        network.input_neurons.append(InputNeurons[ip].copy(network))
    for op in d_op:
        network.output_neurons.append(OutputNeurons[op].copy(network))
    for conn in l_c[0]:
        network.connections[0].append((network.input_neurons[list(d_ip.keys()).index(conn[0])], network.inner_neurons[0][conn[1]], conn[2]))
    for conn in l_c[-2]:
        network.connections[-2].append((network.inner_neurons[-1][conn[0]], network.output_neurons[list(d_op.keys()).index(conn[1])], conn[2]))
    for conn in l_c[-1]:
        network.connections[-1].append((network.input_neurons[list(d_ip.keys()).index(conn[0])], network.output_neurons[list(d_op.keys()).index(conn[1])], conn[2]))
    for l in range(1, BRAIN_DEPTH):
        for conn in l_c[l]:
            network.connections[l].append((network.inner_neurons[l-1][conn[0]], network.inner_neurons[l][conn[1]], conn[2]))


class Network:
    def __init__(self, entity, n_inner: int, n_layers: int , gene: str):
        self.entity = entity
        self.gene = gene
        self.input_neurons = []
        self.output_neurons = []
        self.inner_neurons = [[] for i in range(BRAIN_DEPTH)]
        for l in range(n_layers):
            for i in range(n_inner):
                self.inner_neurons[l].append(Neuron(self, (l+1, i), 0.0, f"inner{l}_{i}"))
        self.connections = [[]] * (BRAIN_DEPTH+2)  # connections[0]: input->output, connections[else]: inner->inner, connections[-1]: inner->output
        # connection = (source, sink, weight)
        interp_gene(self)
        self.color = ""

    def run(self):
        # calculate neurons
        for inp in self.input_neurons:
            inp.get_input()
        for l in range(BRAIN_DEPTH+2):
            conns = self.connections[l]
            sincs = set()
            for conn in conns:
                conn[1].value += conn[0].value * conn[2]
                sincs.add(conn[1])
            if l == BRAIN_DEPTH:
                pass
            elif l == BRAIN_DEPTH + 1:
                pass
            else:
                for sinc in sincs:
                    sinc.value = sigmoid_np(sinc.value)
        # run output
        for out in self.output_neurons:
            out.value = sigmoid(out.value)
            out.act()
        # reset neurons
        for inns in self.inner_neurons:
            for inn in inns:
                inn.reset()

    def makeGraph(self, ax):
        inner_flattened = sum(self.inner_neurons, [])
        conns_flattened = sum(self.connections, [])

        G = nx.MultiDiGraph()
        G.add_nodes_from([(str(neu.id), {'layer': neu.id[0]}) for neu in self.input_neurons+inner_flattened+self.output_neurons])
        G.add_weighted_edges_from([(str(conn[0].id), str(conn[1].id), conn[2]) for conn in conns_flattened])
        layout = nx.multipartite_layout(G, subset_key="layer")

        def weight2color(w):
            color_dict = {
                (-1, -0.75): '#c50707',
                (-0.75, -0.25): '#f23737',
                (-0.25, -0.02): '#d76868',
                (-0.02, 0.02): 'tab:gray',
                (0.02, 0.25): '#92ca86',
                (0.25, 0.75): '#54dd37',
                (0.75, 1): '#0fa03b'
            }
            for interval, color in color_dict.items():
                if interval[0] <= w/MAX_WEIGHT <= interval[1]:
                    return color
            return 'tab:gray'
        
        inp_color_map = ["#073763"]*2 + ["#2917ff"]*2 + ["#0066ff"]*5 + ["#4b46f0"]*4 + ["#245887"]*2 + ["#14868d"]
        out_color_map = ["#9e5050"] + ["#d52c2c"]*4 + ["#f55050"]*2 + ["#cc0011"] + ["#eb572d"] + ["#c82157"]
        edge_color_map = [weight2color(conn[2]) for conn in conns_flattened]

        nx.draw_networkx_nodes(G, layout, nodelist=[str(inp.id) for inp in self.input_neurons], node_color=[inp_color_map[inp.id[1]] for inp in self.input_neurons], alpha=0.9, ax=ax)
        nx.draw_networkx_nodes(G, layout, nodelist=[str(inn.id) for inn in inner_flattened], node_color="tab:gray", alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, layout, nodelist=[str(out.id) for out in self.output_neurons], node_color=[out_color_map[out.id[1]] for out in self.output_neurons], alpha=0.9, ax=ax)

        nx.draw_networkx_edges(G, layout, edgelist=[(str(conn[0].id), str(conn[1].id)) for conn in conns_flattened],  arrows=True, arrowstyle="->", edge_color=edge_color_map, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, layout, {str(inp.id): inp.name for inp in self.input_neurons}, font_size=6, ax=ax)
        nx.draw_networkx_labels(G, layout, {str(inn.id): inn.name for inn in inner_flattened}, font_size=6, ax=ax)
        nx.draw_networkx_labels(G, layout, {str(out.id): out.name for out in self.output_neurons}, font_size=6, ax=ax)
        nx.draw_networkx_edge_labels(G, layout, {(str(conn[0].id), str(conn[1].id)): np.round(conn[2], 3) for conn in conns_flattened}, font_size=3, ax=ax)

        return G

    def printGene(self):
        print(self.gene)


def mutateGene(gene):
    # apply mutation with prob. P_MUTATION for every bit in gene
    new_gene = ''
    for char in gene:
        if rand.random() <= P_MUTATION:
             new_gene += format(rand.randint(0, 15), 'x')
        else:
             new_gene += char
    return new_gene

def randGene():
    # generate random gene
    new_gene = ''
    for i in range(6*BRAIN_SIZE):
        new_gene += format(rand.randint(0, 15), 'x')
    return new_gene


class Entity:
    def __init__(self, x, y):
        self.network = None
        self.x = x
        self.y = y
        self.direction = 0
        self.freq = FRAMES_PER_GEN / 12.5
        self.alive = True

    def reset(self):  
        # reset Entity for new round
        self.network = None
        self.x = 0
        self.y = 0
        if not self.alive:
            self.alive = True

    def inRect(self, coords: tuple, rectangle) -> bool:
        x, y = coords
        interval_x, interval_y = rectangle
        return (interval_x[0] <= x <= interval_x[1]) and (interval_y[0] <= y <= interval_y[1])
    
    def inRectangles(self, rectangles:list) -> bool:
        for rectangle in rectangles:
            if self.inRect((self.x, self.y), rectangle):
                return True
        return False
    
    def move_x(self, delta_x):
        x_prime = self.x + delta_x
        if not (0 <= x_prime <= CANVAS_DIM):
            return
        for rectangle in WALLS:
            if self.inRect((x_prime, self.y), rectangle):
                return
        self.x = x_prime
    
    def move_y(self, delta_y):
        y_prime = self.y + delta_y
        if not (0 <= y_prime <= CANVAS_DIM):
            return
        for rectangle in WALLS:
            if self.inRect((self.x, y_prime), rectangle):
                return
        self.y = y_prime

    def status_check(self):
        # check if Entity is alive
        if self.alive:
            if self.inRectangles(DEAD_ZONE):
                self.alive = False
                entities.remove(self)
                entities_dead.append(self)
                entities.alive -= 1
            else:
                self.alive = True
        else:
            pass

    def run(self):
        if self.alive:
            self.network.run()
        self.status_check()


if __name__ == "__main__":
    print(randGene())
    gene1 = (
        "000000"
        "110111"
        "220222"
        "330333"
        "440444"
        "440555"
        "440555"
        "440555"
    )
    '''
    gene1 = ""
    for i in range(8):
        gene1 += f"0" + hex(8+i)[2:]+ "0000"
    '''
    net1 = Network(None, N_PER_LAYER, BRAIN_DEPTH, gene1)
    # net.run()
    fig = plt.figure(figsize=(4,2), dpi=200)
    ax = fig.add_subplot(1, 1, (1,1))
    net1.makeGraph(ax)
    
    ax.axis("off")
    ax.set_title("net1")
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()
    
    print('done')
    # 테스트할 거 있으면