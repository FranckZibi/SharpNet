using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.GPU;

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
        public List<int> ResourceIds { get; set; } = new List<int> {0};
        public void SetResourceId(int resourceId)
        {
            if (resourceId == int.MaxValue)
            {
                //use multi GPU
                ResourceIds = Enumerable.Range(0, GPUWrapper.GetDeviceCount()).ToList();
            }
            else
            {
                //single resource
                ResourceIds = new List<int> { resourceId };
            }
        }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; set; }

        protected Network BuildEmptyNetwork(string networkName)
        {
            Config.LogFile = networkName + ExtraDescription;
            var network = new Network(Config, ResourceIds);
            network.Description = networkName + ExtraDescription;
            return network;
        }
        public static string GetKerasModelPath(string modelFileName)
        {
            return System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), @".keras\models\", modelFileName);
        }

        public DataAugmentationConfig DA => Config.DataAugmentation;
    }
}
