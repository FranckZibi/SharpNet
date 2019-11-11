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

        #region Data Augmentation

        public double WidthShiftRange { private get; set; }
        public double HeightShiftRange { private get; set; }
        public bool HorizontalFlip { private get; set; }
        public bool VerticalFlip { private get; set; }
        public ImageDataGenerator.FillModeEnum FillMode { private get; set; }

        /// <summary>
        /// The cutout to use in % of the longest length ( = Max(height, width) )
        /// If less or equal to 0 , Cutout will be disabled
        /// </summary>
        public double CutoutPatchPercentage { private get; set; }

        /// <summary>
        /// The alpha coefficient used to compute lambda in CutMix
        /// If less or equal to 0 , CutMix will be disabled
        /// A value of 1.0 will use a uniform random distribution in [0,1] for lambda
        /// </summary>
        public double AlphaCutMix { private get; set; }

        /// <summary>
        /// The alpha coefficient used to compute lambda in Mixup
        /// A value less or equal to 0.0 wil disable Mixup
        /// A value of 1.0 will use a uniform random distribution in [0,1] for lambda
        /// </summary>
        public double AlphaMixup { get; set; }

        /// <summary>
        /// rotation range in degrees, in [0,180] range.
        /// The actual rotation will be a random number in [-_rotationRangeInDegrees,+_rotationRangeInDegrees]
        /// </summary>
        public double RotationRangeInDegrees { private get; set; }

        /// <summary>
        /// Range for random zoom. [lower, upper] = [1 - _zoomRange, 1 + _zoomRange].
        /// </summary>
        public double ZoomRange { private get; set; }
        #endregion

        protected Network BuildEmptyNetwork(string networkName)
        {
            var configLogger = NetworkLogger(networkName);
            Config.Logger = configLogger;
            var network = new Network(Config, DataGenerator(), GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
        }

        public Logger NetworkLogger(string networkName)
        {
            return DisableLogging ? Logger.NullLogger : Utils.Logger(networkName + ExtraDescription);
        }

        private ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchPercentage, AlphaCutMix, AlphaMixup, RotationRangeInDegrees, ZoomRange);
        }
    }
}
