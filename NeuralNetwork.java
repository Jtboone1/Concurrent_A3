import java.util.ArrayList;
import java.util.Collections;

public class NeuralNetwork 
{
	public static ArrayList<Double> inputs;
	public static int max_layer;
	public static ArrayList<ArrayList<Double>> weights;
	public static ArrayList<ArrayList<Double>> neuron_outputs;

	// This will be the outputs to each neuron multiplied by the weight of the branch of that output.
	// We will use this to determine the output of each neuron.
	public static ArrayList<ArrayList<Double>> summed_weights;

	public static void simulate(int L, ArrayList<Double> sim_inputs, ArrayList<ArrayList<Double>> sim_weights)
	{
		inputs = sim_inputs;
		weights = sim_weights;
		max_layer = L;

		summed_weights = new ArrayList<ArrayList<Double>>();
		neuron_outputs = new ArrayList<ArrayList<Double>>();

		// Build summed weight matrix which will just take the outputs to each neuron and multiply them with the
		// branch's weight. Should be the same form as the weights matrix!
		for (int i = 0; i < L + 1; i++)
		{
			// 4 neurons for starting layer with 5 neurons in output side
			if (i == 0)
			{
				for (int j = 0; j < 4; j++)
				{
					ArrayList<Double> layer_weights = new ArrayList<Double>(Collections.nCopies(5, 0.0));
					summed_weights.add(layer_weights);
				}
			}
			// 5 neurons for each hidden layer with 5 neurons passing in output side
			else if (i < L)
			{
				for (int j = 0; j < 5; j++)
				{
					ArrayList<Double> layer_weights = new ArrayList<Double>(Collections.nCopies(5, 0.0));
					summed_weights.add(layer_weights);
				}
			}
			// 3 neurons for final layer with 5 neurons in input side
			else 
			{
				for (int j = 0; j < 5; j++)
				{
					ArrayList<Double> layer_weights = new ArrayList<Double>(Collections.nCopies(3, 0.0));
					summed_weights.add(layer_weights);
				}
			}
		}

		// The idea is to run each layer separately, where every neuron in
		// that layer runs concurrently.
		//
		// Once every neuron in the layer has computed it's output, we move to the next layer. 
		for (int i = 0; i < L + 1; i++)
		{
			try 
			{
				NeuralNetLayer layer = new NeuralNetLayer(i);
				layer.start();

				// We wait for each layer to finish before moving to the next layer.
				layer.join();
			}

			catch (InterruptedException ie)
			{
				ie.printStackTrace();
			}
		}

		System.out.print("\n");

		// Final output to our neural network:
		for (int i = 0; i < NeuralNetwork.neuron_outputs.size(); i++)
		{
			System.out.print("Layer: " + i + " outputs: ");
			for (Double output : NeuralNetwork.neuron_outputs.get(i))
			{
				System.out.print(output + " ");
			}
			System.out.print("\n");
		}

		// Calculate final outputs from last layer by summing final weighted branches.
		ArrayList<Double> final_outputs = new ArrayList<Double>(Collections.nCopies(3, 0.0));
		for (int i = 0; i < 3; i++)
		{
			double output = final_outputs.get(i);
			for (int j = 0; j < 5; j++)
			{
				int size = NeuralNetwork.summed_weights.size();
				output += NeuralNetwork.summed_weights.get(size - 5 + j).get(i);
			}
			final_outputs.set(i, output);
		}


		System.out.print("\n");
		int max_output_index = 0;
		double max_output = -1000000;

		System.out.print("Final Layer Outputs: ");
		for (int i = 0; i < 3; i++)
		{
			max_output_index = final_outputs.get(i) > max_output ? i : max_output_index;
			max_output = final_outputs.get(i) > max_output ? final_outputs.get(i) : max_output;

			System.out.print(final_outputs.get(i) + " ");
		}

		System.out.println("\nLargest index: " + max_output_index);
		System.out.println("Largest output: " + max_output + "\n");
	}
}
