using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wine_Classifier_ANN
{
    public abstract class Core
    {
        public enum ActivationType
        {
            None, 
            Sigmoid,
            ReLu
        }
        public enum LayerType
        {
            None,
            Output,
            Hidden,
            Input
        }
        public static double Sigmoid(double input_number)
        {
            return (double)1 / ((double)1 + Math.Pow(Math.E, -(input_number)));
        }
        public static double Cost(double actual_value, double expected_value)
        {
            return Math.Round((double)Math.Pow(actual_value - expected_value, 2),4) ;
        }
        public static double Cost_Derivative(double actual_value, double expected_value)
        {
            return (double)(2 * (double)(actual_value - expected_value));
        }
        public static double Sigmoid_Derivative(double input_number)
        {
            return (double)input_number * (double)(1 - input_number);
        }
    }
}
