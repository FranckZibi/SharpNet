using SharpNet.DataAugmentation;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global
// ReSharper disable MemberCanBeProtected.Global

namespace SharpNet.Networks
{
    public abstract class NetworkBuilder
    {
        public string ExtraDescription { get; set; } = "";
        public NetworkConfig Config { get; set; }
        public int GpuDeviceId { get; set; }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; set; }
        public bool DisableLogging { private get; set; }

        protected Network BuildEmptyNetwork(string networkName)
        {
            var configLogger = NetworkLogger(networkName);
            Config.Logger = configLogger;
            var network = new Network(Config, GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
        }

        public Logger NetworkLogger(string networkName)
        {
            return DisableLogging ? Logger.NullLogger : Utils.Logger(networkName + ExtraDescription);
        }
        public DataAugmentationConfig DA => Config.DataAugmentation;
    }
}
