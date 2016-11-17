using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetworkFun
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Title = "Neural Network Fun - By Jan Riebe";
            Console.WriteLine("Neural Network Fun - By Jan Riebe\n");
            Console.WriteLine("\nLoad (L) or create (C) network.");
            switch (ChoiceInput("Load or create a new network:", new string[] { "load", "create" }, new char[] { 'l', 'c' }))
            {
                case 0:
                    
                    NetMaker.RecreateLoadedNet(XMLDocLoaderDialogue());
                    break;
                case 1:
                    //TODO dialogue for deciding how to create
                    //TODO layout loading dialogue
                    NetMaker.MakeNetFromLayout();
                    //TODO net qualities dialogue
                    NetMaker.MakeFullyConnectedNet();
                    break;
                default:
                    break;
            }
            //TODO display some information on the nex
            //TODO ask for training or readout
            //TODO loading data
        }

        private static int ChoiceInput(string text, string[] choices, char[] inputs)
        {
            if (inputs.Contains('&'))
                throw new ForbiddenCharException('&');
            Console.WriteLine(text);
            string choiceText = "Please type ";
            for(int i =0; i<choices.Length;i++)
            {
                choiceText += "'"+inputs[i] + "' for " + choices[i] + " ";
            }
            char key = '&';
            while (!inputs.Contains(key))
            {
                Console.WriteLine(choiceText);
                key = Console.ReadKey().KeyChar;
            }
            return Array.IndexOf(inputs, key);
        }

        private static XmlDocument XMLDocLoaderDialogue()
        {
            XmlDocument doc = new XmlDocument();
            doc.PreserveWhitespace = true;

            bool fileFound = false;
            while (!fileFound)
            {
                Console.WriteLine("Enter file URL:");
                fileFound = true;
                try { doc.Load(Console.ReadLine()); }
                catch (System.IO.FileNotFoundException)
                {
                    Console.WriteLine("This file does not exist.");
                    fileFound = false;
                }
            }
            return doc;
        }


    }

    class ForbiddenCharException : Exception
    {
        public ForbiddenCharException(char c)
        {
            Console.WriteLine("ForbiddenCharException: The forbidden character " + c + " was used.");
        }
    }

}
