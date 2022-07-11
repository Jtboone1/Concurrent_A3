import java.util.ArrayList;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.Random;

// Class representing a layer in an artificial neural network
public class NeuralNetLayer extends Thread
{

	// List of neurons in this layer
	ArrayList<Neuron> neurons;
	ArrayList<Double> outputed_weights;
	int layer_id;

	// Each layer thread will have it's own barrier, and the layer thread + neuron threads all need 
	// to complete until we move to the next layer. This is because we need the outputs 
	// of each layer's neurons before we can compute the outputs to the next layer.
	CyclicBarrier layer_barrier;

	// Neuron thread class belongs to NeuralNetLayer in order for each to have 
	// access to their respective barriers.
	public class Neuron extends Thread
	{
		public int neuron_id;
		public double neuron_value;

		public Neuron(int id, double value)
		{
			this.neuron_id = id;
			this.neuron_value = value;
		}

		public void run()
		{
			try 
			{
				// The last hidden layer outputs to 3 neurons, whereas every other layer outputs to 5. 
				int output_neurons = layer_id == NeuralNetwork.max_layer ? 3 : 5;

				// Computes the row in the matrix that correspond to the proper neurons outputs.
				int layer_start_index = layer_id == 0 ? 0 : 4 + (layer_id - 1) * 5; 

				for(int i = 0; i < output_neurons; i++)
				{
					// System.out.println("Neuron: " + neuron_id + " -> " + "Neuron: " + i + " Output: " + output);
					double output = NeuralNetwork.weights.get(layer_start_index + this.neuron_id).get(i) * this.neuron_value;
					NeuralNetwork.summed_weights.get(layer_start_index + this.neuron_id).set(i, output);
				}

				// Thread waits between 5 to 20 ms once all inputs have been computed.
				Thread.sleep(new Random().nextInt(20 - 5) + 5);
				layer_barrier.await();
			}
			catch (BrokenBarrierException | InterruptedException e)
			{
				e.printStackTrace();
			}
		}
	}
	
	// Constructor for neural network layer	
	public NeuralNetLayer(int layer_number)
	{
		// The idea is to wait for each layer to complete it's output,
		// in order to move on to the next layer. We need to wait
		// for each layer thread to complete, and that only happeans
		// when each neuron thread has completed it's output calculation.
		this.layer_id = layer_number;
		this.neurons = new ArrayList<Neuron>();

		// The first layer has 4 neurons so 5 parties (4 neurons threads + first layer thread)
		// and takes in the input files inputs for their output.
		if (layer_number == 0)
		{
			layer_barrier = new CyclicBarrier(5);
			ArrayList<Double> layer_outputs = new ArrayList<Double>();

			for (int i = 0; i < 4; i++)
			{
				Neuron neuron = new Neuron(i, NeuralNetwork.inputs.get(i));
				this.neurons.add(neuron);

				// Save outputs for stream output information.
				layer_outputs.add(NeuralNetwork.inputs.get(i));
			}

			NeuralNetwork.neuron_outputs.add(layer_outputs);
		}
		// Every other layer takes 6 parties (5 neuron threads + 1 layer thread)
		else
		{
			layer_barrier = new CyclicBarrier(6);
			ArrayList<Double> layer_outputs = new ArrayList<Double>();

			for (int i = 0; i < 5; i++)
			{
				int number_inputs = layer_id == 1 ? 4 : 5;
				double input_sum = 0;
				int start_index = layer_id == 1 ? 0 : 4 + (layer_id - 2) * 5;

				for (int j = 0; j < number_inputs; j++)
				{
					input_sum += NeuralNetwork.summed_weights.get(start_index + j).get(i);
				}

				// All neurons in hidden layer take their input sum, which we calculate from the 
				// summed weights matrix, which again, is just the weights of each branch
				// multiplied by the neuron output. Summing these all up
				// gives us the inputs to each hidden layer neuron. We then use these
				// inputs to compute the next layers outputs until we reach the end.

				// Save outputs for stream output information.
				layer_outputs.add(input_sum);

				// One thing to note, this is where the activation function logic is!
				Neuron neuron = new Neuron(i, input_sum < 0 ? 0 : input_sum);
				this.neurons.add(neuron);
			}

			NeuralNetwork.neuron_outputs.add(layer_outputs);
		}

	}
		
	public void run() {
		for (int i = 0; i < neurons.size(); i++)
		{
			// Start threads for neurons in this layer
			neurons.get(i).start();
			try 
			{
				Thread.sleep(5);
			} 
			catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}

		try
		{
			// The barrier + neurons wait for the outputs to complete before joining back with the main thread.
			layer_barrier.await();
			System.out.println("\nLayer " + this.layer_id + " Complete!\nWeighted Branch Calculations: \n");
			for (ArrayList<Double> weights : NeuralNetwork.summed_weights)
			{
				System.out.println(weights);
			}
		}
		catch (BrokenBarrierException | InterruptedException e)
		{
			e.printStackTrace();
		}
	}
}
