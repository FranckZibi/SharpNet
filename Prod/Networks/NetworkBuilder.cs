using System;
using System.Collections.Generic;
using SharpNet.DataAugmentation;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global
// ReSharper disable MemberCanBeProtected.Global

namespace SharpNet.Networks
{
    public abstract class NetworkBuilder
    {
        public enum POOLING_BEFORE_DENSE_LAYER
        {
            /* we'll use an Average Pooling layer of size [2 x 2] before the Dense layer*/
            AveragePooling_2,
            /* we'll use an Average Pooling layer of size [8 x 8] before the Dense layer*/
            AveragePooling_8,

            /* We'll use a Global Average Pooling (= GAP) layer just before the last Dense (= fully connected) Layer
            This GAP layer will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling,

            /* we'll use a Global Average Pooling layer concatenated with a Global Max Pooling layer before the Dense Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n, 2*c, 1, 1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling_And_GlobalMaxPooling,

            /* We'll use a Global Max Pooling layer just before the last Dense (= fully connected) Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalMaxPooling,

            /* no pooling */
             NONE
        };
        public string ExtraDescription { get; set; } = "";
        public NetworkConfig Config { get; set; }
        /// <summary>
        /// resources (both CPU & GPU) available for the network
        /// </summary>
        public List<int> ResourceIds { get; set; }
        public void SetResourceId(int resourceId)
        {
            ResourceIds = new List<int> {resourceId};
        }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; set; }
        public bool DisableLogging { private get; set; }

        protected Network BuildEmptyNetwork(string networkName)
        {
            var configLogger = NetworkLogger(networkName);
            Config.Logger = configLogger;
            var network = new Network(Config, ResourceIds);
            network.Description = networkName + ExtraDescription;
            return network;
        }
        protected static string GetKerasModelPath(string modelFileName)
        {
            return System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), @".keras\models\", modelFileName);
        }



        public Logger NetworkLogger(string networkName)
        {
            return DisableLogging ? Logger.NullLogger : Utils.Logger(networkName + ExtraDescription);
        }
        public DataAugmentationConfig DA => Config.DataAugmentation;
    }
}
