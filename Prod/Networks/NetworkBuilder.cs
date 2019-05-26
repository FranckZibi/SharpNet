using SharpNet.Pictures;

namespace SharpNet.Networks
{
    public abstract class NetworkBuilder
    {
        public string ExtraDescription { get; set; }
        public NetworkConfig Config { get; protected set; }
        public int GpuDeviceId { private get; set; }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; set; }
        public bool DisableLogging { private get; set; }
        #region Data Augmentation
        public double WidthShiftRange { private get; set; }
        public double HeightShiftRange { private get; set; }
        public bool HorizontalFlip { private get; set; }
        public bool VerticalFlip { private get; set; }
        public ImageDataGenerator.FillModeEnum FillMode { private get; set; }
        public int CutoutPatchlength { private get; set; }
        #endregion

        protected Network BuildEmptyNetwork(string networkName)
        {
            Config.Logger = DisableLogging ? Logger.NullLogger : Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, DataGenerator(), GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
        }

        private ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
    }
}