#!/usr/bin/env python

import os
import os.path
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import networkx as nx
import pydot

# random.seed(0)
# np.random.seed(0)

############################################

# https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def to_graph(l) :
    G = nx.Graph()
    for part in l :
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l) :
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it :
        yield last, current
        last = current

def load_graph(dfg_xml, memloc_file, banksize):
    # randomly assign to one bank or the one after it
    var2bank = defaultdict(int)
    with open(memloc_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip the first line "var_name,base_addr"
            var, addr = line.strip().split(",")
            var2bank[var] = round(int(addr)/banksize)

    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(dfg_xml, parser=parser)
    root = tree.getroot()

    DFG = nx.DiGraph()
    backedge_list = []

    attr = defaultdict(dict)
    for nodes in root :
        for nodeelm in nodes :
            if nodeelm.tag == "OP":
                attr[nodes.attrib["idx"]]["op"] = nodeelm.text
            if nodeelm.tag == "BasePointerName":
                attr[nodes.attrib["idx"]]["bank"] = var2bank[nodeelm.text] + random.randint(0,1)
            if nodeelm.tag == 'Outputs' :
                for outputs in nodeelm :
                    DFG.add_edge(nodes.attrib["idx"], outputs.attrib["idx"])
                    if outputs.attrib["nextiter"] == "1" :
                        e = (nodes.attrib["idx"], outputs.attrib["idx"])
                        backedge_list.append(e)

            if nodeelm.tag == 'RecParents' :
                for recparents in nodeelm :
                    e = (nodes.attrib["idx"], recparents.attrib["idx"])
                    backedge_list.append(e)
    print(attr)
    nx.set_node_attributes(DFG, attr)
    np.random.seed(1)

    print(DFG.number_of_nodes())
    return DFG

def plot_dfg(DFG:nx.DiGraph, colors, labels, filename):

    pydot_dfg = pydot.Dot('DFG', graph_type='graph')

    # print("adding nodes to colored dot dfg")
    for i, nodes in enumerate(DFG.nodes) :
        pydot_dfg.add_node(pydot.Node(nodes, fontcolor="white", style="filled",
                                      label=labels[i], fillcolor=colors[i]))

    # print("adding edges to colored dot dfg")
    for edges in DFG.edges :
        pydot_dfg.add_edge(pydot.Edge(edges[0], edges[1]))

    # pydot_dfg.write_png(filename+".png")
    pydot_dfg.write_pdf(filename + ".pdf")


def my_mkdir(dir) :
    try :
        os.makedirs(dir)
    except :
        pass
