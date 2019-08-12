using SharpNet.Pictures;
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global
// ReSharper disable MemberCanBeProtected.Global

namespace SharpNet.Networks
{
    public abstract class NetworkBuilder
    {
        public string ExtraDescription { get; set; }
        public NetworkConfig Config { get; protected set; }
        public int GpuDeviceId { private get; set; }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; protected set; }
        public bool DisableLogging { private get; set; }

        #region Data Augmentation

        public double WidthShiftRange { private get; set; }
        public double HeightShiftRange { private get; set; }
        public bool HorizontalFlip { private get; set; }
        public bool VerticalFlip { private get; set; }
        public ImageDataGenerator.FillModeEnum FillMode { private get; set; }

        /// <summary>
        /// The cutout to use in % of the longest length ( = Max(height, width) )
        /// If less or equal to 0 , cutout will be disabled
        /// </summary>
        public double CutoutPatchPercentage { private get; set; }
        /// <summary>
        /// Should we use CutMix 
        /// </summary>
        public bool CutMix { private get; set; }

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
            Config.Logger = DisableLogging ? Logger.NullLogger : Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, DataGenerator(), GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
        }

        private ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchPercentage, CutMix, RotationRangeInDegrees, ZoomRange);
        }
    }
}
