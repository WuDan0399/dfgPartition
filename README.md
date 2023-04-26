# dfgPartition

**Function:**
1. Allows a big DFG to be partitioned into multiple cluster, with store and load nodes added to pass intermediate result for cross cluster edges.
2. Ensures no cycles among clusters.
3. Ensures a valid memory allocation for intermediate values. The memory node placement can be seen in 'log.txt' and 'memPE_alloc.txt"


**Configs:*
1. Shuffled the candidate ports during searching to avoid congestion in one memory bank. See `map_one()` in main.py
2. To modify the output format, check main.py line 334, function ‘dump_to_xml’.
3. To disable array splitting (randomly choose bank [variable_address\bank_size] or [variable_address\bank_size + 1]), go panorama.py, line 62. Comment out “+ random.randint(0,1)”
4. As the clustering algorithm also has randomness, uncomment panorama.py, line 12-13, to ensure reproducibility. Also, you can comment out and run multiple times to get more results.
