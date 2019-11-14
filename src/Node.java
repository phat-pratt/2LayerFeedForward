import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of
 * nodes. Check the type attribute of the node for details. Feel free to modify
 * the provided function signatures to fit your own implementation
 */

public class Node {
	private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents = null; // Array List that will contain the parents (including the bias
														// node) with weights if applicable

	private double inputValue = 0.0;
	private double outputValue = 0.0;
	private double outputGradient = 0.0;
	private double delta = 0.0; // input gradient

	// Create a node with a specific type
	Node(int type) {
		if (type > 4 || type < 0) {
			System.out.println("Incorrect value for node type");
			System.exit(1);

		} else {
			this.type = type;
		}

		if (type == 2 || type == 4) {
			parents = new ArrayList<>();
		}
	}

	// For an input node sets the input value which will be the value of a
	// particular attribute
	public void setInput(double inputValue) {
		if (type == 0) { // If input node
			this.inputValue = inputValue;
		}
	}
	public double getWeightedSum() {
		double z = 0.0;
		//calculate the weighted 
		for(NodeWeightPair p : parents) {
			z += (p.weight) * (p.node.inputValue); 
		}
		return z;		
	}
	
	/**
	 * Calculate the output of a node. You can get this value by using getOutput()
	 * 
	 * needs to calculate the output activation value at the node if it's type 2 or
	 * 4 The calcuated output needs to be stored in outputValue. Value is determined
	 * by the definition of the activation function (ReLU or Softmax), which depends
	 * on the type of node Type 2 - use ReLU Type 4 - use Softmax
	 * 
	 */
	public void calculateOutput() {
		// if the node is a hidden or output node
		if (type == 2 || type == 4) { // Not an input or bias node
			// use ReLU
			if(type == 2) {
				outputValue = Double.max(0.0, getWeightedSum());
			}
			// use Softmax
			if(type == 4) {
				double z = 0.0;
				//calculate the weighted 
				for(NodeWeightPair p : parents) {
					z += (p.weight) * (p.node.outputValue); 
				}
				double ezj = Math.exp(z);
				// need to normalize ezj
				outputValue = ezj;
				
			}
			
		}
	}

	// Gets the output value
	public double getOutput() {

		if (type == 0) { // Input node
			return inputValue;
		} else if (type == 1 || type == 3) { // Bias node
			return 1.00;
		} else {
			return outputValue;
		}

	}
	
	// Calculate the delta value of a node.
	public void calculateDelta() {
		if (type == 2 || type == 4) {
			// if this is a hidden layer node 
			if(type == 2) {
				double z = 0.0;
				for(NodeWeightPair p : parents) {
					z += (p.weight) * (p.node.outputValue); 
				}
				if(z > 0) {
					delta = 1;
				} else {
					delta = 0;
				}
			}
			// else this is an output node
			else {
				double z = 0.0;
				for(NodeWeightPair p : parents) {
					z += (p.weight) * (p.node.outputValue); 
				}
				if(z > 0) {
					delta = 1;
				} else {
					delta = 0;
				}
			}
		}
	}

	// Update the weights between parents node and current node
	public void updateWeight(double learningRate) {
		if (type == 2 || type == 4) {
			if(type == 2) {
				
			}
			else {
				for(NodeWeightPair p : parents) {
					p.weight = (p.weight - delta);
				}
			}
		}
	}
}
