from itertools import chain
from panorama import *
import re
import sys


label_dict = {0: "comp", 1: "load", 2:"store"}
colordict = {
        0 : 'red',
        1 : 'darkgreen',
        2 : 'blue',
        3 : 'orange',
        4 : 'cyan4',
        5 : 'purple',
        6 : 'orange',
        7 : 'brown',
        8 : 'magenta',
        9 : 'turquoise',
        10 : 'coral1',
        11 : 'black',
        12 : 'sienna',
        13 : 'aquamarine4',
        14 : 'blue4',
        15 : 'lightpink3',
        16 : 'brown',
        17 : 'blueviolet',
        18 : 'maroon4',
        19 : 'deeppink3',
        20 : 'dimgrey',
        21 : 'dodgerblue',
        22 : 'gold4',
        23 : 'silver',
        24 : 'gold',
        25 : 'lightgray',
        26 : 'orange',
        27 : 'brown',
        28 : 'magenta',
        29 : 'gray26',
        30 : 'azure',
        31 : 'black',
        32 : 'sienna',
        33 : 'silver',
        34 : 'gold',
        35 : 'yellow'
    }

def same_dict(a:dict, b:dict):
    if len(a) == 0 or len(b) == 0:
        return False
    for k in a:
        if k not in b:
            continue
        elif a[k]!=b[k]:
            return False
    return True


def count_mem(graph:nx.DiGraph, node_list):
    return len([node for node in node_list if graph.nodes[node]["op"] in ["LOAD", "STORE"]])


def get_cross_cluster_edges(clusters:list, graph:nx.DiGraph):
    node2cluster = {}
    for i, cluster in enumerate(clusters) :
        for node in cluster :
            node2cluster[node] = i

    cross_cluster_edges_node_index = []
    cross_cluster_edges_cluster_index = []
    # cross-cluster completion (add load and store for across cluster edges)
    for src, dest in graph.edges :
        if node2cluster[src] != node2cluster[dest] :
            cross_cluster_edges_node_index.append((src, dest))
            cross_cluster_edges_cluster_index.append((node2cluster[src], node2cluster[dest]))
    return node2cluster, cross_cluster_edges_node_index, cross_cluster_edges_cluster_index


def remove(graph:nx.DiGraph, node:str, node2cluster:dict, clusters:list):
    for src, _ in graph.in_edges(nbunch = node):
        for _, dest in graph.out_edges(nbunch = node):
            graph.add_edge(src,dest)
    graph.remove_node(node)
    clusters[node2cluster[node]].remove(node)
    node2cluster.pop(node)


def remove_redundant_mem(graph, node2cluster:dict, clusters:list):
    remove_list = []
    for src, dest in graph.edges:
        if graph.out_degree[src]==1 and graph.in_degree[dest]==1:
            if (graph.nodes[src]["op"] == "LOAD" and graph.nodes[dest]["op"] == "LOAD") or \
                    (graph.nodes[src]["op"] == "STORE" and graph.nodes[dest]["op"] == "STORE"):
                # two virtual
                if src[:2] == "VR" and dest[:2] == "VR" :
                    remove_list.append(src)  # remove any one is fine
                elif src[:2] != "VR" and dest[:2] != "VR" : # two real nodes
                    continue
                else:  # one virtual one number
                    if src[:2] == "VR":
                        remove_list.append(src)
                    else:
                        remove_list.append(dest)
    for node in remove_list:
        remove(graph, node, node2cluster, clusters)


def memory_allocation(graph:nx.DiGraph, clusters:list, node2cluster:dict, num_banks:int, num_bank_ports:int):
    def same_bank(ports:set, ports2:set):
        return len(ports.intersection(ports2)) > 0

    def same_bank_ports(ports:set): # only allows set of same bank ports as input.
        sample_port = list(ports)[0]
        bank_id = sample_port//num_bank_ports
        return set([bank_id * num_bank_ports + i for i in range(num_bank_ports)])

    def to_bank(ports: set) :
        banks = set()
        for port in ports :
            banks.add(round(port / num_bank_ports))
        return banks

    def bank_to_ports(banks:set):
        ports = []
        for bank in banks:
            ports = ports + [bank*num_bank_ports+i for i in range(num_bank_ports)]
        return set(ports)

    def dependency_propagation(mem_location_dict, f_mem_nodes:list, show:bool) :
        for mem_node in f_mem_nodes:
            # remove for dependent node for mem_node
            for edge in list(graph.in_edges(mem_node)) + list(graph.out_edges(mem_node)) :
                # Only ST->LD has dependency LD->ST can be placed on different banks
                if graph.nodes[edge[0]]["op"] == "STORE" and graph.nodes[edge[1]]["op"] == "LOAD":
                    other_end = edge[0] if edge[1] == mem_node else edge[1]
                    common_banks = to_bank(mem_location_dict[other_end]).intersection(to_bank(mem_location_dict[mem_node]))
                    if len(common_banks) == 0:
                        if show :
                            print(f"Conflict bank for {other_end} and {mem_node}")
                    common_bank_ports = bank_to_ports(common_banks)
                    mem_location_dict[other_end] = mem_location_dict[other_end].intersection(common_bank_ports)
                    mem_location_dict[mem_node] = mem_location_dict[mem_node].intersection(common_bank_ports)
                    if len(mem_location_dict[mem_node]) == 0 :
                        if show:
                            print(f"ERROR: no available space for {mem_node}.")
                        return False
                    if len(mem_location_dict[other_end]) == 0 :
                        if show :
                            print(f"ERROR: no available space for {other_end}.")
                        return False
        return True

    def in_cluster_conflict_avoidance(mem_location_dict, f_mem_nodes: list, show: bool):
        fixed_nodes = set()
        for mem_node in f_mem_nodes :
            if len(mem_location_dict[mem_node]) == 1 :
                fixed_nodes.add(mem_node)
        for fixed_node in fixed_nodes:
            cluster_id = node2cluster[fixed_node]
            for mem_node in mem_nodes[cluster_id] :
                # remove for memory_node in the same clusters
                if mem_node != fixed_node :
                    port_id = list(mem_location_dict[fixed_node])[0]  # seed has a fixed port assigned so it only has one candidate
                    if port_id in mem_location_dict[mem_node] :
                        mem_location_dict[mem_node].remove(port_id)
                        if len(mem_location_dict[mem_node]) == 0 :
                            if show :
                                print(f"ERROR: no available space for {mem_node}.")
                            return False
        return True

    def map_one( mem_location_dict, undetermined:list):
        if len(undetermined) == 0:
            return True, mem_location_dict
        node = undetermined[0]
        candidates = list(mem_location_dict[node])
        random.shuffle(candidates)
        if len(candidates) == 0:
            return False, mem_location_dict
        i = 0
        valid = False
        tmp_mem_location_dict = mem_location_dict.copy() # save a copy as dependency_propagation and in_cluster conflict changes the dict
        while not valid:
            if i == len(candidates):
                # print(f"No available candidate for {node}, fall back")
                return False, mem_location_dict
            mem_location_dict[node] = {candidates[i]}
            # print(f"Map node {node} to {candidates[i]} ({len(undetermined)} waiting)")
            valid = dependency_propagation(mem_location_dict, [node], show=False)
            if not valid:
                mem_location_dict = tmp_mem_location_dict.copy()
                i = i + 1
                continue
            valid = valid and in_cluster_conflict_avoidance(mem_location_dict, [node], show=False)
            if not valid:
                mem_location_dict = tmp_mem_location_dict.copy()
                i = i + 1
                continue
            sorted_nodes = sorted(undetermined[1:], key= lambda x: len(mem_location_dict[x]))
            valid_map, mapping = map_one(mem_location_dict.copy(), sorted_nodes)
            valid = valid and valid_map
            if valid:
                mem_location_dict = mapping.copy()
            else:
                mem_location_dict = tmp_mem_location_dict.copy()
            i = i+1
        return True, mem_location_dict

    feasible = True
    # 1. initialize the memory allocation
    mem_loc_dict = defaultdict(set)  # stores an integer: idx = bank_id * num_bank_ports + port_id
    seeds = set()  # fixed bank at the beginning
    mem_nodes = [[node for node in cluster if graph.nodes[node]["op"] in ["LOAD", "STORE"]] for cluster in clusters]
    for mem_nodes_in_cluster in mem_nodes:
        for mem_node in mem_nodes_in_cluster:
            if 'bank' in graph.nodes[mem_node]:
                next_port = len([x for x in seeds.intersection(mem_nodes_in_cluster)
                                 if graph.nodes[x]["bank"] == graph.nodes[mem_node]["bank"]])
                mem_loc_dict[mem_node].add(graph.nodes[mem_node]['bank']*num_bank_ports + next_port)
                seeds.add(mem_node)
            else:
                mem_loc_dict[mem_node] = set([i for i in range(num_banks*num_bank_ports)])

    # determined dependencies and conflicts
    prev_mem_loc_dict = {}
    flatten_mem_nodes = list(chain.from_iterable(mem_nodes))
    while not same_dict(mem_loc_dict, prev_mem_loc_dict):
        prev_mem_loc_dict = mem_loc_dict
        feasible = dependency_propagation(mem_loc_dict, flatten_mem_nodes, show=True)
        feasible = feasible and in_cluster_conflict_avoidance(mem_loc_dict, flatten_mem_nodes, show=True)

    # map ops to ports for non-determined ops
    non_determined_ports = {k:mem_loc_dict[k] for k in mem_loc_dict if len(mem_loc_dict[k])>1} # remove if leads to failure
    sorted_keys = sorted(non_determined_ports.keys(), key=lambda x: len(x[1]))  # start with the one with the least candidates
    # sorted_keys = sorted(non_determined_ports.keys(), key=lambda x: int(x.split("_")[-1]))
    valid_map, mapping = map_one(mem_loc_dict.copy(), sorted_keys)
    # valid_map, mapping = map_one(0, mem_loc_dict.copy(), list(non_determined_ports.keys()))
    if valid_map:
        mem_loc_dict = mapping
    feasible = feasible and valid_map
    return mem_loc_dict, feasible


def augment(cross_cluster_edges:list,
            node2cluster:dict,
            clusters:list,
            graph:nx.DiGraph):
    index = 0
    cross_edge_dict = defaultdict(set)
    for src, dest in cross_cluster_edges:
        if graph.nodes[src]["op"] == 'LOAD' and graph.nodes[dest]["op"] == 'STORE':
            continue  # inherently good for partition
        else:
            cross_edge_dict[src].add(dest)
    for src in cross_edge_dict:
        src_index = index
        graph.add_node(f"VR_ST_{src_index}", op="STORE")
        index = index + 1
        clusters[node2cluster[src]].append(f"VR_ST_{src_index}")
        node2cluster[f"VR_ST_{src_index}"] = node2cluster[src]

        graph.add_edge(src, f"VR_ST_{src_index}")

        for dest in cross_edge_dict[src]:
            # update graph: replace the original edge with three new edges and two new nodes
            dest_index = index
            graph.add_node(f"VR_LD_{dest_index}", op="LOAD")
            index = index + 1
            clusters[node2cluster[dest]].append(f"VR_LD_{dest_index}")
            node2cluster[f"VR_LD_{dest_index}"] = node2cluster[dest]

            graph.add_edge(f"VR_ST_{src_index}", f"VR_LD_{dest_index}")
            graph.add_edge(f"VR_LD_{dest_index}", dest)
            graph.remove_edge(src, dest)

    # remove_redundant_mem(graph, node2cluster, clusters)

def no_cluster_cycle(cross_cluster_edges:list):
    # dict for cluster-node mapping
    cluster_graph = nx.DiGraph(cross_cluster_edges)
    cycles_across_clusters = nx.recursive_simple_cycles(cluster_graph)
    if len(cycles_across_clusters) != 0 :
        print("[IMPORTANT] There is across-cluster cycle")
        for i, cycle in enumerate(cycles_across_clusters):
            print(f"\tcycle {i}:")
            for hop in cycle:
                print(f"\t\tcluster {hop}")
                print(clusters[hop])
        return False
    return True


def process(clusters:list, graph:nx.DiGraph, n_total_pe:int, num_bank:int, num_bank_ports:int):
    n_mem_pe = num_banks * num_bank_ports

    node2cluster, cross_cluster_edges_node_index,\
        cross_cluster_edges_cluster_index = get_cross_cluster_edges(clusters, graph)


    print("\n\nClustering result before processing")
    print(clusters)
    print("\n\nCluster size before processing")
    print([len(cluster) for cluster in clusters])

    # augment graph with LD|ST insertion
    augment(cross_cluster_edges_node_index, node2cluster, clusters, graph)

    print("\n\nClustering result after processing")
    print(clusters)
    print("\n\nCluster size after processing")
    print([len(cluster) for cluster in clusters])

    # across-cluster acyclic check
    if not no_cluster_cycle(cross_cluster_edges_cluster_index):
        return {}, {}, False

    # in-cluster check
    for i, cluster in enumerate(clusters):
        if len(cluster) > n_total_pe:
            print(f"[Overflow Op] cluster {i} has {len(cluster)} operations")
            return {}, {}, False
        mem_in_cluster = count_mem(graph, cluster)
        if mem_in_cluster > n_mem_pe:
            print(f"[Overflow MEM Op] cluster {i} has {mem_in_cluster} memory operations")
            return {}, {}, False

    mem_loc, feasible = memory_allocation(graph, clusters, node2cluster.copy(), num_bank, num_bank_ports)

    return node2cluster, mem_loc, feasible


def dfg_partition(graph:nx.DiGraph, num_cluster:int):
    # initial clustering
    clusters = nx.community.greedy_modularity_communities(graph, cutoff=num_cluster)  #
    return clusters

def dump_to_xml(out_prefix:str, dfg_filename:str, graph:nx.DiGraph, clusters:list, node2cluster:dict, mem_loc:dict, folder):
    mem_loc_file = open(os.path.join(folder, "memPE_alloc.txt"), 'w')
    mem_loc_file.write("node_id\tport_id\tcontent\n")
    original_file = open(dfg_filename, 'r')
    lines = original_file.readlines()
    original_file.close()
    out_files = {}
    for cluster_id in range(len(clusters)):
        out_files[cluster_id] = open(os.path.join(folder, out_prefix+f"_{cluster_id}.xml"), 'w')
    it_line = 0
    node_id = -1
    cluster_id = -1
    # DFG nodes in original xml
    while it_line < len(lines):
        line = lines[it_line]
        if line[:5] == "<Node":
            node_id = re.findall("[0-9]+",line.strip().split()[1])[0]
            cluster_id = node2cluster[node_id]
            if graph.nodes[node_id]["op"] in ["LOAD", "STORE"]:
                # line_with_mem_port = line[:-2] + f' PORT="{list(mem_loc[node_id])[0]}">\n'
                # line_with_mem_port = line[:-2]
                out_files[cluster_id].write(line)
                bank_id = graph.nodes[node_id]["bank"]
                mem_loc_file.write(f"{node_id}\t{list(mem_loc[node_id])[0]}\tcontent in bank{bank_id}\n")
            else:
                out_files[cluster_id].write(line)

            it_line = it_line + 1
        elif line[:9] == '\t<Output ':
            # dump the edges
            for _, dest in graph.out_edges(node_id):
                if node2cluster[dest] == node2cluster[node_id] :
                    out_files[cluster_id].write(f'\t<Output idx="{dest}" nextiter="0" type="I1"/>\n')
            # skip output section
            while lines[it_line][:9] == '\t<Output ':
                it_line = it_line + 1
        elif line[:8] == '\t<Input ':
            # dump the edges
            for src, _ in graph.in_edges(node_id):
                if node2cluster[src] == node2cluster[node_id]:  # don't write cross cluster edges
                    out_files[cluster_id].write(f'\t<Input idx="{src}" />\n')
            # skip input section
            while lines[it_line][:8] == '\t<Input ':
                it_line = it_line + 1
        else:  # OP, Inputs, Outputs tag, <RecParents>, </Node>. In this case, no <RecParents> needed
            if cluster_id != -1 and node_id != -1:
                out_files[node2cluster[node_id]].write(line)
            it_line = it_line + 1

    virtual_BasePointerName = {}
    # newly added virtual DFG nodes
    for node in graph.nodes:
        if node[:2] == "VR":
            cluster_id = node2cluster[node]
            # node id
            fake_ASAP = len(clusters[cluster_id])
            if graph.nodes[node]["op"] in ["LOAD", "STORE"]:
                out_files[node2cluster[node]].write(
                    f'<Node idx="{node}" ASAP="{fake_ASAP}" ALAP="{fake_ASAP}" BB="fake" CONST="1">\n')
            else:
                out_files[node2cluster[node]].write(
                    f'<Node idx="{node}" ASAP="{fake_ASAP}" ALAP="{fake_ASAP}" BB="fake" CONST="1">\n')
            # operation info
            out_files[node2cluster[node]].write(
                f'<OP>{graph.nodes[node]["op"]}</OP>\n')
            # BasePointerName for virtual load and store node  # todo: get <BasePointerName size="256">
            if graph.nodes[node]["op"] in ["LOAD", "STORE"] :
                # store the intermediate value, use the parent id (real) or parent's former node as input
                former_node = "null"
                in_edges = graph.in_edges(node)
                srcs = [x for x, y in in_edges]
                if len(srcs) != 1 :
                    print("[ERROR] A virtual store has more than one parent node.")
                src = srcs[0]
                if src[:2] == "VR":
                    if src not in virtual_BasePointerName:
                        # This should not happen as the virtual nodes are created along dependencies.
                        # Their id implicitly embeds the dependency.
                        print(f"[ERROR] A virtual node {node} is written to file before its parent {src}. Missing BasePointerName")
                    else:
                        former_node = virtual_BasePointerName[src]
                else:
                    virtual_BasePointerName[node] = src
                    former_node = src

                out_files[node2cluster[node]].write(f'<BasePointerName size = "256"> {former_node}</BasePointerName>\n')
                mem_loc_file.write(f"{node}\t{list(mem_loc[node])[0]}\t{former_node}\n")
            # begin inputs
            out_files[node2cluster[node]].write(
                f'<Inputs>\n')
            for src, _ in graph.in_edges(node):
                if node2cluster[src] == node2cluster[node] :
                    out_files[cluster_id].write(f'\t<Input idx="{src}" />\n')
            out_files[node2cluster[node]].write(
                f'</Inputs>\n')
            # begin outputs
            out_files[node2cluster[node]].write(
                f'<Outputs>\n')
            for _, dest in graph.out_edges(node) :
                if node2cluster[dest] == node2cluster[node] :
                    out_files[cluster_id].write(f'\t<Output idx="{dest}" nextiter="0" type="I1"/>\n')
            out_files[node2cluster[node]].write(
                f'</Outputs>\n')
            out_files[node2cluster[node]].write(
                f'<RecParents>\n</RecParents>\n</Node>\n\n')

    for _, file in out_files.items():
        file.close()   # Good habit of closing the file ;)


if __name__ == '__main__':
    dfg_filename = "mod.xml"
    memloc_file = "jpeg_fdct_islow_INNERMOST_LN1_mem_alloc.txt"
    ###### PE array configs #######
    n_total_pe = 16
    num_cluster = 10  #  10
    num_banks = 4
    num_bank_ports = 2
    bank_size = 1024
    ###############################

    feasible = False
    while not feasible:
        graph = load_graph(dfg_filename, memloc_file, bank_size)
        clusters = dfg_partition(graph, num_cluster)
        clusters = [list(cluster) for cluster in clusters]

        # post-process
        node2cluster, mem_loc, feasible = process(clusters, graph, n_total_pe, num_banks, num_bank_ports)

        num_cluster = num_cluster + 1

    if len(clusters)>24:
        sys.exit()

    labels = []
    node_color = []
    for node in graph.nodes :
        tmp_op = graph.nodes[node]["op"]
        label = f"{node}_{tmp_op}"
        if node in mem_loc:
            if mem_loc[node] == [x for x in range(num_bank_ports*num_banks)]:
                label = label + "_[0-8]"
            else:
                label = label + f"_{mem_loc[node]}_c{node2cluster[node]}"
        labels.append(label)
        node_color.append(colordict[node2cluster[node]])

    plot_dfg(graph, node_color, labels, f"partition_{len(clusters)}")

    print("\n\n[MEMORY NODE PLACEMENT] nodeid: portid")
    print(mem_loc)

    folder = f"{len(clusters)}_cluster"
    i = 0
    while os.path.exists(folder + f"_{i}"):
        i = i + 1
    folder = folder + f"_{i}"
    os.mkdir(folder)
    print(f"\nOutput to {folder}")

    os.rename(f"partition_{len(clusters)}.png", os.path.join(folder, f"partition_{len(clusters)}.png"))
    dump_to_xml("partitioned_DFG", dfg_filename, graph, clusters, node2cluster, mem_loc, folder)
