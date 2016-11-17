using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetworkFun
{
    
    abstract class NeuronalNode
    {

    }
    
    class Neuron : NeuronalNode
    {
        /*
        Neurons consist of a bounch of inputs, including their weights and a bunch of links to outputs.
        The inputs are called parents, the outputs children.
        For each input the Neuron stores a weight. The weights of the outputs are stored by the children.
        Each Neuron has a Delta which is a sum of the weighted deltas of all children.
        Each Neuro has an output which is the sum of its weighed inputs 
        A Neuron also has a threshold weight, that is like an input weight attached to -1.
        The neurons in the last layer have Neurons of the derived type TrainingDesiredValue as children.
        */ 
        private Dictionary<NeuronalNode, double> parents;     // The double value is the weight.
        private List<NeuronalNode> children = new List<NeuronalNode>();
        private double delta = 0;
        protected double output = 0;
        private double thresholdWeight;

        public Neuron(List<NeuronalNode> assignParents, List<double> assignWeights, double assignThresholdWeight)
        {            
            //Assigning the parents and the weights.
            #region
            if (assignWeights == null)
            {
                parents = new Dictionary<NeuronalNode, double>();
                foreach (NeuronalNode n in assignParents)
                {

                    parents.Add(n, 0);
                    if (n.GetType() == typeof(Neuron))
                        ((Neuron)n).AddNeuronAsChild(this);
                }
                AssignRandomWeights();
            }
            else
            {
                parents = new Dictionary<NeuronalNode, double>();
                    
                int i = 0;
                foreach (NeuronalNode n in assignParents)
                {
                    try
                    {
                        parents.Add(n, assignWeights[i]);
                        if (n.GetType() == typeof(Neuron))
                            ((Neuron)n).AddNeuronAsChild(this);
                    }
                    catch (IndexOutOfRangeException e)
                    {
                        Console.WriteLine("Error during Neuron initialisation: More parents then weights given.\n-> " + e.Message);
                    }
                    i++;
                }
                thresholdWeight = assignThresholdWeight;
            }
            #endregion
            
        }

        //Assigns all weights, including threshold weight a random value between 0.0 and 1.0
        public void AssignRandomWeights()
        {
            foreach (Neuron n in parents.Keys.ToArray())
            {
                parents[n] = new Random().NextDouble();
            }
            thresholdWeight = new Random().NextDouble();
        }

        //Returns the delta of the Neuron.
        public double GetDelta()
        {
            return delta;
        }

        //Returns the weight between a specified input neuron and this neuron.
        public double GetParentWeight(NeuronalNode parent)
        {
            if (parent == null)
                return thresholdWeight;
            return parents[parent];
        }

        //Resturns the value of this neuron after summing all weighted inputs and passing the value throught the sigmoid function.
        public double GetOutput()
        {
            return output;
        }

        //Calculates a new Delta by calling the Delta function in NeuralMath. Used when propagating back.
        public void CalculateDelta()
        {
            delta = NeuralMath.Delta(output, children, this);
        }

        //Calculate new input and threshold weights by calling the NewWeight and NewThresholdWeight function in Neural Math. Used when propagating forward.
        public void CalculateWeights()
        {
            foreach(Neuron n in parents.Keys.ToArray())
            {
                parents[n] = NeuralMath.NewWeight(parents[n], n, this);
            }
            thresholdWeight = NeuralMath.NewThresholdWeight(thresholdWeight, this);
        }

        public void AddNeuronAsChild(NeuronalNode newChild)
        {
            children.Add(newChild);
        }
                       
    }

    //Nodes that get attached to the last layer of neurons and contains the desired value for training.
    class TrainingDesiredValue : NeuronalNode
    {
        private double desiredValue;

        public void SetDesiredValue(double value)
        {
            desiredValue = value;
        }

        public double GetDesiredValue()
        {
            return desiredValue;
        }
    }

    //Nodes that get attached as parents of the first layer 
    class InputValue : NeuronalNode
    {
        private double output;
        public void SetValue(double value)
        {
            output = value;
        }
    }

    static class Backpropagator
    {
        /*
        In backpropagation the neurons first get assigned new deltas calculating from children to parents,
        then new weights, calculating from parents to children.

        In each step first the training data gets assigned to the input and the output neurons.
        For this SetValue gets called on all inputValues.
        Then SetDesiredValue gets called on all trainingDesiredValues.
        After that for each layer children to parents CalculateDelta gets called on all neurons in that layer.
        Finally for each layer parents to children CalculateWeights gets called on all neurons in that layer.
        Then the step gets repeated as often as defined in the taining data.
        Once the training data has been repeated as defined, new training data gets loaded and the process starts again.
        */
        private static List<List<double>> inputValues;
        private static List<List<double>> trainingDesiredValues;
        private static List<InputValue> inputNodes;
        private static List<List<Neuron>> neuronLayers;
        private static List<TrainingDesiredValue> outputNodes;

        public static void SetNet(List<InputValue> input, List<List<Neuron>> net, List<TrainingDesiredValue> output)
        {
            inputNodes = input;
            neuronLayers = net;
            outputNodes = output;
        }

        public static void SetTrainingSet(List<List<double>> inputs, List<List<double>> desired)
        {
            inputValues = inputs;
            trainingDesiredValues = desired;
        }
    }

    //Creates the Neurons, connects and initialises them. Then passes them on to the Backpropagator.
    static class NetMaker
    {
        private static List<List<Neuron>> net;
        private static List<InputValue> inputNodes;
        private static List<TrainingDesiredValue> outputNodes;

        public static void MakeFullyConnectedNet(List<int> layerWidths)
        {
            //TODO
        }
        public static void MakeNetFromLayout(XmlDocument netLayoutXML)
        {
            //TODO
        }
        public static void RecreateLoadedNet(XmlDocument netXML)
        {
            /*
            The net gets build up by loading the neurons layer by layer from parents to children.
            This way a neuron can get its parents assigned directly when it gets created.
            A node gets its children when they call on their parents.
            Weights get loaded as well.

            First the input and desired output layer get build,
            then the neurons inbetween.
            The first layer of neurons has always one input per neuron, the last one output per neuron.
            */
            XmlNode root = netXML.DocumentElement;
            XmlNodeList layers = root.ChildNodes;
            
            inputNodes = new List<InputValue>();
            for(int j=0;j< layers[0].ChildNodes.Count; j++)
            {
                inputNodes.Add(new InputValue());
            }

            outputNodes = new List<TrainingDesiredValue>();
            for (int j = 0; j < layers[layers.Count-1].ChildNodes.Count; j++)
            {
                outputNodes.Add(new TrainingDesiredValue());
            }

            net = new List<List<Neuron>>();           
            int i = 0;
            foreach (XmlNode layer in layers)
            {
                List<Neuron> neurons = new List<Neuron>();
                XmlNodeList neuronNodes = layer.ChildNodes;
                int j = 0;
                foreach (XmlNode neuronNode in neuronNodes)
                {
                    //Creating the neuron. The neuron is fully connected, except for the last layer that still needs children.
                    Neuron newNeuron = XMLNodeToNeuron(neuronNode, i);
                    neurons.Add(newNeuron);
                    //Assigning the children of the last layer.
                    newNeuron.AddNeuronAsChild(outputNodes[j]);
                    j++;
                }
                net.Add(neurons);
                i++;
            }

            //Passing the net to the Backpropagator.
            Backpropagator.SetNet(inputNodes, net, outputNodes);
        }
        
        //Reads all attributes of a neuron node and creates a fully connected Neuron object from them.
        private static Neuron XMLNodeToNeuron(XmlNode node,int currentLayer)
        {
            
            List<NeuronalNode> parents = new List<NeuronalNode>();
            List<double> weights = new List<double>();
            double thresholdWeight = 0;

            XmlNodeList childNodes = node.ChildNodes;
        
            foreach(XmlNode parent in childNodes)
            {
                if (parent.Name == "parents")
                {
                    foreach (XmlNode index in parent.ChildNodes)
                    {
                        if (index.Name == "index")
                        {
                            if (currentLayer>0)
                                parents.Add(net[currentLayer - 1][int.Parse(index.InnerText)]);
                            else
                                parents.Add(inputNodes[int.Parse(index.InnerText)]);
                        }
                        else if (index.Name == "weight")
                            weights.Add(double.Parse(index.InnerText));
                    }
                }
                else if (parent.Name == "thresholdWeight")
                    thresholdWeight = double.Parse(parent.InnerText);
            }
            
            return new Neuron(parents, weights,thresholdWeight);
        }
        
    }

    //This class wraps all the neccessary math to do backpropagation.
    static class NeuralMath
    {
        public static double rateConstant;
        /*
        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
        private double DerivativeSigmoid(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        //You are trying to find the maximum of a performance function P, by changing the weights associated with neurons,.
        private double PerformanceFunction(double desired, double observed)
        {
            return -Math.Pow(desired - observed,2)/2;
        }

        private double DerivativePerformanceFunction(double desired, double observed)
        {
            return desired - observed;
        }
        */

        //Calculates a new weight based on the desired change in the performance function, contained in the rate constant and the delta of this neuron, and the output value of the parent node.
        public static double NewWeight(double oldweight, Neuron parentNeuron, Neuron thisNeuron)
        {
            return oldweight + rateConstant * /*input*/ parentNeuron.GetOutput() * thisNeuron.GetDelta();//Delta(desired, observed, children, thisNeuron);
        }

        //Calculates a new threshold weight based on the desired change in the performance function, contained in the rate constant and the delta of this neuron. Does not need the parent nodes, because it is allways attached to the value -1.
        public static double NewThresholdWeight(double oldweight, Neuron thisNeuron)
        {
            return oldweight + rateConstant * -1 * thisNeuron.GetDelta();
        }

        //Sets the constant that defines the step size on the performance function.
        public static void SetRateConstant(double rc)
        {
            rateConstant = rc;
        }


        /*
        private double DeltaFinal(double desired, double observed)
        {
            //return DerivativePerformanceFunction() * DerivativeSigmoid();
            return (desired - observed) * observed * (1 - observed);
        }
        */

        //Calculates a neurons delta. For the final layer it compares the desired with the observed value, using the derivative of the performance function. For all other neurons, it calculates the delta from the sum of all childrens deltas.
        public static double Delta(double observed, List<NeuronalNode> children, Neuron thisNeuron)
        {
            if (children[0].GetType()==typeof(TrainingDesiredValue))
            {
                double desired = ((TrainingDesiredValue)children[0]).GetDesiredValue();
                return (desired - observed) * observed * (1 - observed);
            }

            double sumOfChildren = 0;
            foreach(Neuron n in children)
            {
                sumOfChildren += n.GetParentWeight(thisNeuron) * n.GetDelta();
            }
            return observed * (1 - observed) * sumOfChildren;
        }
        /*
        //I calculate new weights from how I want the results to change in order to optimise the Performance function.
        private static void GradientAccent()
        {


            
            for all layers except the last
            Δw = α × i × δli
            δli = oli(1 − oli) × SUMj[wli →rj × δrj]
            

            for the final layer
            Δw = α × i × δfk
            δfk = ofk(1 − ofk) × (dk − ofk)

            That is, you computed change in a neuron’s w, in every layer,
            by multiplying α times the neuron’s input times its δ.
            The δ is determined for all but the final layer in terms of
            the neuron’s output and all the weights that connect that output 
            to neurons in the layer to the right and the δs associated with those right-side neurons. 
            The δ for each neuron in the final layer is determined only 
            by the output of that neuron and by the difference between 
            the desired output and the actual output of that neuron. 

            So I could calculate all δ from right to left as they dont contain inputs,
            store them in the neuron.
            In this calculation I use the values in the layer to the right.
            And then calculate weights left to right.
            I repeat this until the performance function is maximised.
            (Do I run it over the same test input and output though, or over alternating?
            Watch the lecture again!)

            α is a value that the user chooses.

            A neuron holds it's weight, a δ and links to it's input and output neurons.
            The first layer is connected to inputNeurons which extend Neuron, but have a given output value (that becomes the input for the next layer) and don't calculate weights.
            The last layer is consists of lastNeurons which extend Neuron, but have a given output value. They calculate weights but no delta. They handle outputting.
            
        }
    */
    }
}
