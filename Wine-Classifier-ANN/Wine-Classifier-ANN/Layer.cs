using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Wine_Classifier_ANN.Core;

namespace Wine_Classifier_ANN
{
    public class Layer
    {
        private ActivationType ActivationType;
        public LayerType LayerType;
        public Neuron[] Neurons;
        public Layer()
        {
            
        }
        public Layer(int number_of_neurons, ActivationType activationType, LayerType layerType)
        {
            Neurons = new Neuron[number_of_neurons];
            for (int i = 0; i < number_of_neurons; i++)
            {
                Neurons[i] = new Neuron(activationType, layerType);
            }
            ActivationType = activationType;
            LayerType = layerType;

        }
    }
}
