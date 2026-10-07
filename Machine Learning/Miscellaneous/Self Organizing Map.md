## Self-Organizing Map (SOM)
* **Definition:** An unsupervised neural network algorithm used for clustering and visualizing high-dimensional data. It maps input data onto a lower-dimensional (typically 2D) grid while preserving the topological properties of the data.
![](https://upload.wikimedia.org/wikipedia/commons/3/35/TrainSOM.gif)
### Key Points
#### Competitive Learning
* **Definition:** Competitive learning is the core mechanism by which SOM assigns input data to output neurons. It’s a "winner-takes-all" process where neurons compete to represent an input data point based on similarity (typically measured by distance).
* **Process:**
	1. **Input Presentation**: An input data point (a vector, e.g., principal components) is presented to the SOM.
	2. **Distance Calculation**: For each output neuron in the 2D grid, the Euclidean distance (or another metric) between its weight vector and the input vector is computed.
	3. **Find the BMU:** Find the node in the map closest to the input data point.
	4. **Update Nodes:** For each node, update its vector by pulling it closer to the input vector.
#### Neighborhood Function
* **Definition:** The neighborhood function extends the learning beyond the winner to nearby neurons in the 2D grid, enabling the SOM to preserve the topological structure of the input data.
### Criticisms
* The algorithm is derived from heuristic ideas, rather than statistical principles, leading to several issues:
	1. Cluster number is restricted by the number of neurons.
	2. Due to unclear clustering boundaries, the final clustering varies depending on initial neuron weights.
	3. Some datapoints are unrepresented.
## Training Data Splitting Method (TDSM)
Splitting the training samples to discover ‘‘missing’’ neurons whose members can be perfectly represented without modifying the original network’s algorithmic structure and internal operations.
