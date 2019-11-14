import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        System.out.println("num input nodes: " + inputNodeCount + "\nNum  output nodes: " + outputNodeCount);
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
    	double max = 0.0;
    	int ans = 0;
    	for(int i = 0; i < inputNodes.size() - 1; i++) {
    		inputNodes.get(i).setInput(instance.attributes.get(i));
    	}
    	for(int i = 0; i < hiddenNodes.size() - 1; i++) {
    		hiddenNodes.get(i).calculateOutput();
    	}
    	max = 0.0;
    	for (int i = 0; i < outputNodes.size(); i++) {
    		outputNodes.get(i).calculateOutput();
    		double o = outputNodes.get(i).getOutput();
    		
    		if(o > max) {
    			max = o;
    			ans = i;
    		}
    	}
        return ans;
    }

    public void forwardPass(Instance instance) {
    	for(int i = 0; i < inputNodes.size() - 1; i++) {
    		inputNodes.get(i).setInput(instance.attributes.get(i));
    	}
    	for(int i = 0; i < hiddenNodes.size() - 1; i++) {
    		hiddenNodes.get(i).calculateOutput();
    	}
    	for (int i = 0; i < outputNodes.size(); i++) {
    		outputNodes.get(i).calculateOutput();
    	}
    }
    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
    	
    	for(int i = 0; i < maxEpoch; i ++) {
			// 1a. shuffle 
			Collections.shuffle(trainingSet, random);
			// 1. predict 
			double totalLoss = 0.0;
			for(Instance instance : trainingSet) {
				// forward pass. At each hidden unit use the ReLU activation function
				forwardPass(instance);
				
				// backward pass. for each hidden unit, and output unit, compute delta x
				for(Node h : hiddenNodes) {
					h.calculateDelta();
				}
				for(Node o : outputNodes) {
					o.calculateDelta();
				}
				// update weights
				for(Node h : hiddenNodes) {
					h.updateWeight(learningRate);
				}
				for(Node o : outputNodes) {
					o.updateWeight(learningRate);
				}
				
				// calculate error (Cross Entropy Loss) at each output unit
			}
			for(Instance instance : trainingSet) {
				totalLoss += loss(instance);
			}

			// 2 . update the weights
			totalLoss = totalLoss/trainingSet.size();
			System.out.println("Epoch: " + i + ", Loss: " + String.format("%.3e", totalLoss));
    	}
        // TODO: add code here
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
    	double totalLoss = 0.0;
    	for(int i = 0; i < instance.classValues.size(); i++) {
    		if(instance.classValues.get(i) != 0) {
    			return -Math.log(outputNodes.get(i).getOutput());
    		}
    	}
    	return -1.0;
    }
}
