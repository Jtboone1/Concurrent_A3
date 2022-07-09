import java.io.*;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args)
    {

        // Pass in "new" to generate a new set of inputs and weights
        if (args.length > 0 && args[0].equals("new"))
        {
            System.out.print("HERE");
            RandomInputGenerator input_generator = new RandomInputGenerator("inputs.txt", 4, 0.0, 5.0);
            input_generator.GenerateRandomInputsFile();

            RandomWeightGenerator weight_generator = new RandomWeightGenerator("weights.txt", 20, -0.5, 5.0, 0.1);
            weight_generator.GenerateRandomWeightFile();
        }

        ArrayList<Double> inputs = new ArrayList<Double>();
        ArrayList<ArrayList<Double>> weights = new ArrayList<ArrayList<Double>>();
        int L = 0;

        try 
        {

            // Read input file
            File input_file = new File("./inputs.txt");
            FileReader input_fr = new FileReader(input_file);
            BufferedReader input_br = new BufferedReader(input_fr);

            String[] doubles;
            String line = input_br.readLine();
            doubles = line.split(",");

            for (String num : doubles)
            {
                inputs.add(Double.parseDouble(num));
            }

            // Read weights file
            File weight_file = new File("./weights.txt");
            FileReader weight_fr = new FileReader(weight_file);
            BufferedReader weight_br = new BufferedReader(weight_fr);

            // This builds the weight matrix from the weights.txt file
            while ((line = weight_br.readLine()) != null)
            {
                doubles = line.split(",");
                if (doubles.length == 1)
                {
                    L = Integer.parseInt(doubles[0]);
                }
                else
                {

                    ArrayList<Double> layer_weights = new ArrayList<Double>();
                    for (String string_num : doubles)
                    {
                        layer_weights.add(Double.parseDouble(string_num));
                    }

                    // Each neuron will have it's own weight array output, which we will use
                    // to calculate each hidden layer input / output.
                    weights.add(layer_weights);
                }
            }

            weight_br.close();
            input_br.close();
        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
        }

        NeuralNetwork.simulate(L, inputs, weights);
    }
}
