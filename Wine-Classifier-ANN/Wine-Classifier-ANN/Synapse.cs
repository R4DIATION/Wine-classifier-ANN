using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wine_Classifier_ANN
{
    public class Synapse
    {
        static Random rnd = new Random();
        public double Weight;
        public Neuron InputNeuron;
        public Neuron OutputNeuron;
        public Synapse(Neuron input, Neuron output)
        {
            this.InputNeuron = input;
            this.OutputNeuron = output;
            this.Weight = rnd.NextDouble() * 2 - 1;
        }
    }
}
