# dfgPartition

**Overview**
1. [code](https://github.com/WuDan0399/dfgPartition/blob/e9bf5ebce506a7af20c1c1cfb9c8f98bdb778cd0/panorama.py#L40) read xml file and memory allocation file to generate `nx.Digraph` with node attributes: 'bank' and 'op'. `panorama.py` -> `load_graph()`
2. [code](https://github.com/WuDan0399/dfgPartition/blob/e9bf5ebce506a7af20c1c1cfb9c8f98bdb778cd0/main.py#L611) partition the graph with `num_cluster` clusters  `main.py` -> `dfg_partition()`
3. [code](https://github.com/WuDan0399/dfgPartition/blob/e9bf5ebce506a7af20c1c1cfb9c8f98bdb778cd0/main.py#L615) add virtual load and store nodes for cross-cluster edges. For multiple edges sharing the same source vertex, only one virtual store node will be added. 
4. [code](https://github.com/WuDan0399/dfgPartition/blob/e9bf5ebce506a7af20c1c1cfb9c8f98bdb778cd0/main.py#L618) Repeatedly try merge two small clusters and generate memory allocation (to port) for all load\store nodes, virtual nodes included. `main.py` -> `merge()`
5. In 2-4, cluster size, memory nodes in one cluster and cluster-level cycles are checked, if fails, increase the `num_cluster` and repeat 2-4. In 4, if there is no valid memory node mapping, increase the `num_cluster` and repeat 2-4.
6. For a valid partition, add select node for each cluster (cluster inherently with select node are excluded), connect the select node with all load and store nodes, virtual included. `main.py` -> `add_select()`,
7.  [code](https://github.com/WuDan0399/dfgPartition/blob/e9bf5ebce506a7af20c1c1cfb9c8f98bdb778cd0/main.py#L477) dump each cluster to xml file.  `main.py` -> `dump_to_xml()`

**Function:**
1. Allows a big DFG to be partitioned into multiple cluster, with store and load nodes added to pass intermediate result for cross cluster edges.
2. Ensures no cycles among clusters.
3. Ensures a valid memory allocation for intermediate values. The memory node placement can be seen in 'log.txt' and 'memPE_alloc.txt"


**Configs:**
1. Shuffled the candidate ports during searching to avoid congestion in one memory bank. See `map_one()` in main.py
2. To modify the output format, check main.py line 334, function ‘dump_to_xml’.
3. To disable array splitting (randomly choose bank [variable_address\bank_size] or [variable_address\bank_size + 1]), go panorama.py, line 62. Comment out “+ random.randint(0,1)”
4. As the clustering algorithm also has randomness, uncomment panorama.py, line 12-13, to ensure reproducibility. Also, you can comment out and run multiple times to get more results.
