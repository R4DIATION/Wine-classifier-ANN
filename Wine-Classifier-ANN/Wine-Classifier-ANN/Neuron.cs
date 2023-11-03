using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Wine_Classifier_ANN.Core;

namespace Wine_Classifier_ANN
{
    public class Neuron
    {
        static Random rnd = new Random();
        private double activation;
        public double Activation
        {
            get { return activation; }
            set { activation = value; }
        }
        private double weightedSum;
        public double WeightedSum
        {
            get { return weightedSum; }
        }
        private ActivationType ActivationFunction;
        private double bias;
        public double Bias
        {
            get { return bias; }
            set { bias = value; }
        }
        private Layer ConnectedLayer;
        public Synapse[] synapses;
        private LayerType _containingLayerType;

        public LayerType ContainingLayerType
        {
            get { return _containingLayerType; }
            set { _containingLayerType = value; }
        }

        public Neuron(ActivationType activationType, LayerType ContainingLayerType)
        {
            ActivationFunction = activationType;
            this.ContainingLayerType = ContainingLayerType;
            Activation = 0.0;
            Bias = 1.0;
        }
        public void SetConnectedNeurons(Layer layer)
        {
            this.synapses = new Synapse[layer.Neurons.Length];
            
            for (int i = 0; i < synapses.Length; i++)
            {
                synapses[i] = new Synapse(layer.Neurons[i], this);
            }
            
            
            ConnectedLayer = layer;
        }
        public void CalculateActivation()
        {
            if (this.ActivationFunction != ActivationType.None)
            {
                double WeightedSum = 0.0;
                for (int i = 0; i < this.synapses.Length; i++)
                {
                    WeightedSum += (double)this.synapses[i].Weight * (double)this.synapses[i].InputNeuron.Activation;
                }
                WeightedSum = WeightedSum + Bias;
                this.weightedSum = WeightedSum;
                this.Activation = Math.Round(Sigmoid(WeightedSum), 4);

            }
            else
            {
                double WeightedSum = 0.0;
                for (int i = 0; i < this.synapses.Length; i++)
                {
                    WeightedSum += (double)this.synapses[i].Weight * (double)this.synapses[i].InputNeuron.Activation;
                }
                WeightedSum = WeightedSum + Bias;
                this.weightedSum = WeightedSum;
                this.Activation = WeightedSum;
            }
        }
        public void SetInputValue(double inputValue)
        {
            this.Activation = inputValue;
        }
    }
}
