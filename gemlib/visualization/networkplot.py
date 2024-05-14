import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def plot_network(nodes, labels, filename, dirpath):
    colors = ['red', 'green', 'blue', 'purple', 'sienna', 'gold', 'deeppink', 'magneta',
              'drakorange', 'lime', 'deepskyblue', 'lightsteelblue', 'salmon', 'grey']
    d = {}
    b = {}
    l = zip(labels, nodes)
    for k, v in l:
        d.setdefault(k, []).append(v)
        b.update({v: k})
    plt.figure(figsize=(10, 10))
    b.update({k: k for k in d}) # adding vertexes
    g = nx.Graph(d)
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, cmap=matplotlib.colors.ListedColormap(colors[0:len(d)]), node_size=700,
                     font_color='k', node_color=[b[n] for n in g.nodes()])

    # for labeling outside the node
    offset = 10
    pos_labels = {}
    keys = pos.keys()
    for key in keys:
        x, y = pos[key]
        pos_labels[key] = (x, y + offset)
    nx.draw_networkx_labels(g, pos=pos_labels, fontsize=2)

    plt.axis('off')
    path = dirpath + '_' + filename + '.png'
    plt.savefig(path, dpi=600)
    print('file {0} saved.'.format(path))
