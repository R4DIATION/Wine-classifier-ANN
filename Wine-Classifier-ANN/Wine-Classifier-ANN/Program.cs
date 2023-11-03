using System.Runtime.CompilerServices;

namespace Wine_Classifier_ANN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork network = new NeuralNetwork(16, 1);

            string[] Menu = new string[]
            {
                "1. Training",
                "2. Testing",
                "3. Exit"
            };
            Menu.DrawMenu();
            int indexer = 0;
            string input = "";
            while (input != "3")
            {
                Console.Clear();
                Menu.DrawMenu();
                Console.Write("Your choice: ");
                input = Console.ReadLine();
                if (input == "1")
                {
                    List<string> list = File.ReadAllLines("winequality-white.csv").Skip(1).Select(x => x).ToList();

                    int index = 0;
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (list.Count == index)
                        {
                            index = 0;
                        }
                        else
                        {
                            network.SetInputLayer(list[index]);
                            network.ConnectLayers();
                            network.CalculateActivations();
                            network.CalculateCost();
                            index++;
                        }
                    }
                    Console.WriteLine("Press any key to continue!");
                    Console.ReadKey();
                }
                else if (input == "2")
                {
                    List<string> list = File.ReadAllLines("winequality-red.csv").Skip(1).Select(x => x).ToList();

                    int index = 0;
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (list.Count == index)
                        {
                            index = 0;
                        }
                        else
                        {
                            network.SetInputLayer(list[index]);
                            network.ConnectLayers();
                            network.CalculateActivations();
                            network.CalculateCost();
                            index++;
                        }
                    }
                    Console.WriteLine("Press any key to continue!");
                    Console.ReadKey();
                }
                
            }



        }
        
    }
}