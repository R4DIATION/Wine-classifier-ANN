using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Wine_Classifier_ANN.Core;

namespace Wine_Classifier_ANN
{
    public class NeuralNetwork
    {
        private double ExpectedValue = 0;
        private const int number_of_output_neurons = 1;
        private const int number_of_input_params = 11;
        private const double LearningRate = 0.001;
        private Layer[] layers;
        private double CostPerRowOfData = 0;
        private int LearningCycles = 0;
        public NeuralNetwork(int number_of_neurons_in_first_layer, int number_of_layers = 1)
        {
            layers = new Layer[number_of_layers + 2];
            layers[0] = new Layer(number_of_input_params, ActivationType.None, LayerType.Input);
            layers[1] = new Layer(number_of_neurons_in_first_layer, ActivationType.Sigmoid, LayerType.Hidden);
            layers[2] = new Layer(number_of_output_neurons, ActivationType.None, LayerType.Output);
        }
        public NeuralNetwork(int number_of_neurons_in_first_layer,int number_of_neurons_in_second_layer, int number_of_layers)
        {
            layers = new Layer[number_of_layers + 2];
            layers[0] = new Layer(number_of_input_params, ActivationType.None, LayerType.Input);
            layers[1] = new Layer(number_of_neurons_in_first_layer, ActivationType.Sigmoid, LayerType.Hidden);
            layers[2] = new Layer(number_of_neurons_in_second_layer, ActivationType.Sigmoid, LayerType.Hidden);
            layers[3] = new Layer(number_of_output_neurons, ActivationType.None, LayerType.Output);
        }
        public void SetInputLayer(string line)
        {
            string[] helper = line.Split(';');

            for (int i = 0; i < helper.Length- 1; i++)
            {
                layers[0].Neurons[i].SetInputValue(double.Parse(helper[i], System.Globalization.NumberStyles.AllowDecimalPoint));
            }
            this.ExpectedValue = double.Parse(helper.Last(), System.Globalization.NumberStyles.AllowDecimalPoint);

        }
        public void ConnectLayers()
        {
            for (int i = 1; i < layers.Length; i++)
            {
                for(int j = 0; j < layers[i].Neurons.Length; j++) 
                {
                    layers[i].Neurons[j].SetConnectedNeurons(layers[i - 1]);
                }
            }
        }
        public void CalculateActivations()
        {
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    layers[i].Neurons[j].CalculateActivation();
                }
                
            }
        }
        public void CalculateCost()
        {
            CostPerRowOfData = (double)Cost(layers.Last().Neurons[0].Activation, this.ExpectedValue);
            if (CostPerRowOfData > 0.01)
            {
                Backpropagate();
                LearningCycles++;
            }
            else
            {
                Console.WriteLine($"Cost per row of data: {CostPerRowOfData}, Output value: {layers.Last().Neurons.First().Activation}, Expected Value: {this.ExpectedValue}");
                Backpropagate();
            }
        }
        public double GetCost() => this.CostPerRowOfData;
        private void Backpropagate()
        {
            Neuron outputNeuron = layers.Last().Neurons.First();

            double derivative_of_cost_with_respect_to_predicted_output = 2 * (outputNeuron.Activation - this.ExpectedValue);

            double[] derivative_of_predicted_output_with_respect_to_weight = new double[layers[1].Neurons.Length];

            double derivative_of_predicted_output_with_respect_to_bias = 1;
            double[] derivative_of_cost_with_respect_to_previous_weights = new double[layers[1].Neurons.Length];
            for (int i = 0; i < layers[1].Neurons.Length; i++)
            {
                derivative_of_predicted_output_with_respect_to_weight[i] = layers[1].Neurons[i].Activation;
                derivative_of_cost_with_respect_to_previous_weights[i] = derivative_of_cost_with_respect_to_predicted_output * derivative_of_predicted_output_with_respect_to_weight[i];
            }

            double derivative_of_cost_with_respect_to_previous_bias = derivative_of_cost_with_respect_to_predicted_output * derivative_of_predicted_output_with_respect_to_bias;

            for (int i = 0; i < layers.Last().Neurons.Length; i++)
            {
                for (int j = 0; j < layers.Last().Neurons[i].synapses.Length; j++)
                {
                    double weight = layers.Last().Neurons[i].synapses[j].Weight;
                    layers.Last().Neurons[i].synapses[j].Weight = weight - LearningRate * derivative_of_cost_with_respect_to_previous_weights[j];
                }
                double prevBias = layers.Last().Neurons[i].Bias;
                layers.Last().Neurons[i].Bias = prevBias - LearningRate * derivative_of_cost_with_respect_to_previous_bias;
            }


            double[] derivative_of_predicted_output_with_respect_to_hidden_activation = new double[layers[1].Neurons.Length];
            for (int i = 0; i < layers[1].Neurons.Length; i++)
            {
                derivative_of_predicted_output_with_respect_to_hidden_activation[i] = layers[2].Neurons[0].synapses[i].Weight;
            }

            double[] derivative_of_hidden_activation_with_respect_to_hidden_weighted = new double[layers[1].Neurons.Length];
            for (int i = 0; i < layers[1].Neurons.Length; i++)
            {
                derivative_of_hidden_activation_with_respect_to_hidden_weighted[i] = layers[1].Neurons[i].Activation * (1 - layers[1].Neurons[i].Activation);
            }

            int index = 0;
            double[] derivative_of_hidden_weight_with_respect_to_hidden_weighted = new double[layers[1].Neurons.Length * layers[0].Neurons.Length];
            double[] derivative_of_hidden_weight_with_respect_to_cost = new double[layers[1].Neurons.Length * layers[0].Neurons.Length];
            for (int i = 0; i < layers[1].Neurons.Length; i++)
            {
                for (int j = 0; j < layers[1].Neurons[i].synapses.Length; j++)
                {
                    derivative_of_hidden_weight_with_respect_to_hidden_weighted[index] = layers[1].Neurons[i].synapses[j].InputNeuron.Activation;
                    derivative_of_hidden_weight_with_respect_to_cost[i] = derivative_of_hidden_weight_with_respect_to_hidden_weighted[index] * derivative_of_predicted_output_with_respect_to_hidden_activation[i] * derivative_of_hidden_activation_with_respect_to_hidden_weighted[i] * derivative_of_cost_with_respect_to_predicted_output;
                    index++;
                }
            }
            int index2 = 0;


            for (int i = 0; i < layers[1].Neurons.Length; i++)
            {
                for (int j = 0; j < layers[1].Neurons[i].synapses.Length; j++)
                {
                    double hidden_weight = layers[1].Neurons[i].synapses[j].Weight;
                    layers[1].Neurons[i].synapses[j].Weight = hidden_weight - LearningRate * derivative_of_hidden_weight_with_respect_to_cost[i];
                    index2++;
                }
            }



        }
    }
}
