using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNet.Networks
{
    public class Network
    {
        #region fields
        private readonly ImageDataGenerator _imageDataGenerator;
        private BackwardPropagationManager _backwardPropagationManager;
        public NetworkConfig Config { get; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { private get; set; } = "";
        private readonly Stopwatch _spInternalFit = new Stopwatch();
        private readonly Stopwatch _swUpdateWeights;
        private readonly Stopwatch _swPredictTraining;
        private readonly Stopwatch _swPredictNotTraining;
        private readonly Stopwatch _swBackwardPropagation;
        private readonly Stopwatch _swCreateInputForEpoch;
        private readonly Stopwatch _swComputeLossAndAccuracy;
        private readonly Stopwatch _swComputeLoss;
        private readonly Stopwatch _swComputeAccuracy;
        private Tensor _yPredictedBufferForEntireBatch;
        private Tensor _yExpectedBufferForEntireBatch;
        private Tensor bufferComputeAccuracy;
        private Tensor bufferComputeLoss;
        private readonly int _gpuDeviceId;

        private readonly List<EpochData> _epochsData;
        private readonly DateTime _timeStampCreation = DateTime.Now;
        private string UniqueId => (string.IsNullOrEmpty(Description) ? "Network" : Utils.ToValidFileName(Description)) + "_" + _timeStampCreation.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture);
        public bool UseGPU => _gpuDeviceId != -1;
        #endregion
        public GPUWrapper GpuWrapper { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="config"></param>
        /// <param name="imageDataGenerator"></param>
        /// <param name="gpuDeviceId">
        /// if -1
        ///     run the network on CPU (no GPU usage)
        /// else
        ///     run the network on the GPU with device Id 'gpuDeviceId'
        /// </param>
        /// <param name="epochData"></param>
        public Network(NetworkConfig config, ImageDataGenerator imageDataGenerator, int gpuDeviceId, List<EpochData> epochData = null)
        {
            Config = config;
            _imageDataGenerator = imageDataGenerator??ImageDataGenerator.NoDataAugmentation;
            _epochsData = epochData ?? new List<EpochData>();
            _gpuDeviceId = gpuDeviceId;
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(gpuDeviceId) : null;
            if (config.ProfileApplication)
            {
                _swUpdateWeights = new Stopwatch();
                _swPredictTraining = new Stopwatch();
                _swPredictNotTraining = new Stopwatch();
                _swBackwardPropagation = new Stopwatch();
                _swCreateInputForEpoch = new Stopwatch();
                _swComputeLossAndAccuracy = new Stopwatch();
                _swComputeLoss = new Stopwatch();
                _swComputeAccuracy = new Stopwatch();
            }
            CreateLogDirectoryIfNeeded();
            _backwardPropagationManager = new BackwardPropagationManager(this);
        }

        public List<EpochData> EpochDatas =>  _epochsData;

        #region Transfer Learning
        public DenseLayer ChangeNumberOfCategoriesForTransferLearning(int newNumberOfCategories)
        {
            if (Layers.Count>=2 && Layers[Layers.Count - 1] is ActivationLayer && Layers[Layers.Count-2] is DenseLayer)
            {
                var lambdaL2Regularization = ((DenseLayer) Layers[Layers.Count - 2])._lambdaL2Regularization;
                var activationFunctionType = ((ActivationLayer) Layers[Layers.Count - 1]).ActivationFunction;
                //we remove the last layer (ActivationLayer)
                Layers.Last().RemoveFromNetwork().Dispose();
                //we remove the Dense layer
                Layers.Last().RemoveFromNetwork().Dispose();
                //we add new (not initialized) DenseLayer & ActivationLayer
                Output(newNumberOfCategories, lambdaL2Regularization, activationFunctionType);
                _epochsData.Clear();
                return ((DenseLayer) Layers[Layers.Count - 2]);
            }
            throw new NotImplementedException();
        }

        /// <summary>
        /// Last layer containing weights but frozen (not trainable)
        /// </summary>
        /// <returns></returns>
        public Layer LastFrozenLayer()
        {
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var l = Layers[i];
                if (l != null && !l.Trainable && l.TotalParams > 0)
                {
                    return l;
                }
            }
            return null;
        }
        #endregion

        /// <summary>
        /// Clone the current network
        /// </summary>
        /// <param name="newGpuWrapper">
        /// if null the network will be cloned for CPU usage
        /// if not null, the network will be cloned to work on the GPU embedded in 'newGpuWrapper'
        /// </param>
        /// <returns></returns>
        public Network Clone(GPUWrapper newGpuWrapper)
        {
            var clonedNetworkGpuDeviceId = newGpuWrapper?.DeviceId ?? -1;
            var clonedNetwork = new Network(Config, _imageDataGenerator, clonedNetworkGpuDeviceId, new List<EpochData>(_epochsData));
            clonedNetwork.Description = Description;
            foreach (var l in Layers)
            {
                clonedNetwork.Layers.Add(l.Clone(clonedNetwork));
            }
            return clonedNetwork;
        }

        private void CreateLogDirectoryIfNeeded()
        {
            if (!string.IsNullOrEmpty(Config.LogDirectory) && !Directory.Exists(Config.LogDirectory))
            {
                Directory.CreateDirectory(Config.LogDirectory);
            }
        }
        public string DeviceName() { return GpuWrapper?.DeviceName(); }

        public void Dispose()
        {
            LogDebug("Before clearing memory: " + GpuWrapper?.MemoryInfo());
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect();
            Layers.ForEach(x => x?.Dispose());
            Layers.Clear();
            _epochsData.Clear();

            bufferComputeAccuracy?.Dispose();
            bufferComputeAccuracy = null;

            bufferComputeLoss?.Dispose();
            bufferComputeLoss = null;

            _yPredictedBufferForEntireBatch?.Dispose();
            _yPredictedBufferForEntireBatch = null;

            _yExpectedBufferForEntireBatch?.Dispose();
            _yExpectedBufferForEntireBatch = null;

            _backwardPropagationManager?.Dispose();
            _backwardPropagationManager = null;

            GpuWrapper?.Reset();

            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect();
            LogDebug("After clearing memory: " + GpuWrapper?.MemoryInfo());
        }

        /// <summary>
        /// Compares the 'this' network with the 'other' network a,n,d write a test report in the 'errors' output field
        /// </summary>
        /// <param name="other">the network to compare with the 'this' network</param>
        /// <param name="errors">field where the report results will be stored</param>
        /// <returns>true if the 2 networks are the same, else if a difference has been found</returns>
        public bool Equals(Network other, out string errors)
        {
            var id = Description;
            errors = "";
            const double epsilon = 1e-5;
            var equals = true;
            equals &= Utils.Equals(Description, other.Description, id + ":Description", ref errors);
            equals &= Config.Equals(other.Config, epsilon, id, ref errors);
            equals &= _imageDataGenerator.Equals(other._imageDataGenerator, epsilon, id, ref errors);
            equals &= Utils.Equals(other._gpuDeviceId, _gpuDeviceId, id, ref errors);
            equals &= Utils.Equals(Layers.Count, other.Layers.Count, id + ":Layers.Count", ref errors);
            for (int i = 0; i < Math.Min(Layers.Count, other.Layers.Count); ++i)
            {
                equals &= Layers[i].Equals(other.Layers[i], epsilon, id + ":Layers["+i+"]", ref errors);
            }
            return equals;
        }

        #region network construction: adding layers
        public Network Input(int channelCount, int h, int w)
        {
            Layers.Add(new InputLayer(channelCount, h, w, this));
            return this;
        }
        
        public Network SimpleRnnLayer(int xLength, int aLength, int yLength, bool returnSequences)
        {
            Debug.Assert(Layers.Count >= 1);
            var simpleRnnLayer = new SimpleRnnLayer(xLength, aLength, yLength, returnSequences, this);
            Layers.Add(simpleRnnLayer);
            return this;
        }
        public Network Dense(int n_x, double lambdaL2Regularization)
        {
            Debug.Assert(Layers.Count >= 1);
            var fullyConnectedLayer = new DenseLayer(n_x, lambdaL2Regularization, this);
            Layers.Add(fullyConnectedLayer);
            return this;
        }

        public Network Convolution_BatchNorm(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return Convolution(filtersCount, f, stride, padding, lambdaL2Regularization, true)
                .BatchNorm();
        }
        public Network Convolution_BatchNorm_Activation(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Convolution_BatchNorm(filtersCount, f, stride, padding, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        public Network BatchNorm_Activation(cudnnActivationMode_t activationFunction)
        {
            return BatchNorm().Activation(activationFunction);
        }
        public Network BatchNorm_Activation_Convolution(cudnnActivationMode_t activationFunction, int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, bool useBias)
        {
            return 
                BatchNorm()
                .Activation(activationFunction)
                .Convolution(filtersCount, f, stride, padding, lambdaL2Regularization, useBias);
        }
        public Network AddLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex)
        {
            Layers.Add(new AddLayer(previousIdentityLayerIndex, previousResidualLayerIndex, this));
            Debug.Assert(Layers[previousIdentityLayerIndex].SameOutputShape(Layers[previousResidualLayerIndex]));
            return this;
        }
        public Network ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2)
        {
            Layers.Add(new ConcatenateLayer(previousLayerIndex1, previousLayerIndex2, this));
            return this;
        }
        //add a shortcut from layer 'AddSumLayer' to current layer, adding a Conv Layer if necessary (for matching size)
        public Network Shortcut_IdentityConnection(int startOfBlockLayerIndex, int filtersCount, int stride, double lambdaL2Regularization)
        {
            int previousResidualLayerIndex = Layers.Last().LayerIndex;

            var sameInputAndOutputShapeInBlock = Layers.Last().SameOutputShape(Layers[startOfBlockLayerIndex]);
            if (sameInputAndOutputShapeInBlock)
            {
                Layers.Add(new AddLayer(startOfBlockLayerIndex, previousResidualLayerIndex, this));
            }
            else
            {
                //we need to add a convolution layer to make correct output format
                Convolution(filtersCount, 1, stride, 0, lambdaL2Regularization, true, startOfBlockLayerIndex);
                int convLayerIdInIdentityBlock = Layers.Last().LayerIndex;
                Layers.Add(new AddLayer(convLayerIdInIdentityBlock, previousResidualLayerIndex, this));
                Debug.Assert(Layers[convLayerIdInIdentityBlock].SameOutputShape(Layers[previousResidualLayerIndex]));
            }
            return this;
        }
        public Network Convolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, bool useBias)
        {
            return Convolution(filtersCount, f, stride, padding, lambdaL2Regularization, useBias, Layers.Count - 1);
        }
        public Network Convolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, bool useBias, int previousLayerIndex)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(filtersCount, f, stride, padding, lambdaL2Regularization, useBias, previousLayerIndex, this));
            return this;
        }
        public Network Dropout(double dropProbability)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new DropoutLayer(dropProbability, this));
            return this;
        }
        public Network Activation(cudnnActivationMode_t activationFunction)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ActivationLayer(activationFunction, this));
            return this;
        }
        public Network MaxPooling(int poolingSize, int poolingStride)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingStride,
                this));
            return this;
        }
        public Network AvgPooling(int poolingSize, int poolingStride)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, poolingSize,
                poolingStride, this));
            return this;
        }
        public Network GlobalAvgPooling()
        {
            var lastLayerShape = Layers.Last().OutputShape(1);
            var lastLayerShapeHeight = lastLayerShape[2];
            //We ensure that weight and height are the same
            Debug.Assert(lastLayerShapeHeight == lastLayerShape[3]);
            int poolingSize = lastLayerShapeHeight;
            int poolingStride = lastLayerShapeHeight;
            return AvgPooling(poolingSize, poolingStride);
        }
        public Network BatchNorm(double momentum = 0.99, double epsilon = 1e-5)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new BatchNormalizationLayer(momentum, epsilon, this));
            return this;
        }
        public Network Dense_Activation(int n_x, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        public Network Dense_DropOut_Activation(int n_x, double lambdaL2Regularization, double dropOut, cudnnActivationMode_t activationFunction)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Dropout(dropOut)
                .Activation(activationFunction);
        }
        public Network Output(int n_x, double lambdaL2Regularization, cudnnActivationMode_t activationFunctionType)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Activation(activationFunctionType);
        }
        public Network Flatten()
        {
            Debug.Assert(Layers.Count >= 1);
            var flattenLayer = new FlattenLayer(this);
            Layers.Add(flattenLayer);
            return this;
        }
        #endregion

        //= ForwardPropagation
        public Tensor Predict(Tensor X, bool isTraining)
        {
            (isTraining ? _swPredictTraining : _swPredictNotTraining)?.Start();
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            ((InputLayer)Layers[0]).Set_y(X);
            foreach (var l in Layers.Skip(1))
            {
                l.ForwardPropagation(isTraining);
            }
            (isTraining ? _swPredictTraining : _swPredictNotTraining)?.Stop();
            return Layers.Last().y;
        }
        public void BackwardPropagation(Tensor yExpected)
        {
            _swBackwardPropagation?.Start();
            var yPredicted = Layers.Last().y;
            Debug.Assert(yPredicted != null);
            Debug.Assert(yExpected.SameShape(yPredicted));

            //we compute: dyPredicted = (1.0 / categoryCount)*(yPredicted - yExpected)
            var dyPredicted = _backwardPropagationManager.dyOfLastLayer;
            yPredicted.CopyTo(dyPredicted);
            var categoryCount = yPredicted.Shape[1];
            var multiplier = Layers.Last().IsSigmoidActivationLayer()? (1f / categoryCount) :1f;
            dyPredicted.AddTensor(-multiplier, yExpected, multiplier);

            _backwardPropagationManager.BackwardPropagation();
            _swBackwardPropagation?.Stop();
        }

        public string Summary()
        {
            return Layers.Any(x => x.PreviousLayers.Count >= 2) ? SummaryWithConnectedTo() : SummaryWithoutConnectedTo();
        }
        private string SummaryWithoutConnectedTo()
        {
            const int firstColumnWidth = 29;
            const int secondColumnWidth = 26;
            const int thirdColumnWidth = 10;
            var line0 = new string('_', firstColumnWidth+ secondColumnWidth+ thirdColumnWidth);
            var line1 = new string('=', line0.Length);
            string result = "";
            if (!string.IsNullOrEmpty(Description))
            {
                result += "Network Name: " + Description+ Environment.NewLine;
            }
            result += line0 + Environment.NewLine;
            result += "Layer (Type)                 Output Shape              Param #" + Environment.NewLine;
            result += line1 + Environment.NewLine;
            foreach (var l in Layers)
            {
                var outputShape = Utils.ShapeToStringWithBacthSize(l.OutputShape(1));
                var firstColumn = l.SummaryName()+" ("+l.Type()+")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth-1);
                }
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{l.TotalParams,-thirdColumnWidth}").TrimEnd() + Environment.NewLine;
                result += (l.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams;
            return result;
        }
        private string SummaryWithConnectedTo()
        {
            const int firstColumnWidth = 32;
            const int secondColumnWidth = 21;
            const int thirdColumnWidth = 12;
            const int forthColumnWidth = 33;
            var line0 = new string('_', firstColumnWidth + secondColumnWidth + thirdColumnWidth+ forthColumnWidth);
            var line1 = new string('=', line0.Length);
            string result = "";
            if (!string.IsNullOrEmpty(Description))
            {
                result += "Network Name: " + Description + Environment.NewLine;
            }
            result += line0 + Environment.NewLine;
            result += "Layer (type)                    Output Shape         Param #     Connected to" + Environment.NewLine;
            result += line1 + Environment.NewLine;
            foreach (var l in Layers)
            {
                var outputShape = Utils.ShapeToStringWithBacthSize(l.OutputShape(1));
                var firstColumn = l.SummaryName() + " (" + l.Type() + ")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth - 1);
                }
                var previousLayers = l.PreviousLayers.OrderBy(x=>x.LayerIndex).ToList();
                var firstPreviousLayer = (previousLayers.Count == 0 ? "" : previousLayers[0].SummaryName()+"[0][0]");
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{l.TotalParams,-thirdColumnWidth}{firstPreviousLayer,-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                for (int i = 1; i < previousLayers.Count; ++i)
                {
                    result += ($"{"",-(firstColumnWidth+secondColumnWidth+thirdColumnWidth)}{previousLayers[i].SummaryName() + "[0][0]",-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                }
                result += (l.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams;
            return result;
        }
        private void ResetWeights()
        {
            foreach (var l in Layers)
            {
                if (l != null && l.Trainable)
                {
                    l.ResetWeights();
                }
            }
        }
        public override string ToString()
        {
            var result = Summary() + Environment.NewLine;
            result += Utils.MemoryBytesToString(BytesByBatchSize) + "/batchSize+" + Utils.MemoryBytesToString(BytesIndependantOfBatchSize);
            return result;
        }

        private int MaxMiniBatchSize()
        {
            var freeMemoryInBytes = UseGPU?(ulong)GpuWrapper.AvailableMemoryInBytes() : Utils.AvailableRamMemoryInBytes();
            int maxMiniBatchSize = MaxMiniBatchSize(BytesByBatchSize, BytesIndependantOfBatchSize, freeMemoryInBytes);
            LogDebug("Max MiniBatchSize=" + maxMiniBatchSize + " (free memory=" + Utils.MemoryBytesToString(freeMemoryInBytes) + ")");
            return maxMiniBatchSize;
        }

        //TODO add tests
        private static int MaxMiniBatchSize(ulong bytesByBatchSize, ulong bytesIndependantOfBatchSize, ulong freeMemoryInBytes)
        {
            freeMemoryInBytes -= bytesIndependantOfBatchSize;
            freeMemoryInBytes = (80* freeMemoryInBytes)/100;
            ulong miniBatchSize = 1;
            while ( (2UL * miniBatchSize * bytesByBatchSize) < freeMemoryInBytes)
            {
                miniBatchSize *= 2;
            }
            return (int)miniBatchSize;
        }

        public Optimizer GetOptimizer(int[] weightShape, int[] biasShape)
        {
            switch (Config.OptimizerType)
            {
                case Optimizer.OptimizationEnum.Adam: return new Adam(this, Config.Adam_beta1, Config.Adam_beta2, Config.Adam_epsilon, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD: return new Sgd(this, Config.SGD_momentum, Config.SGD_usenesterov, weightShape, biasShape);
                default: return VanillaSgd.Instance;
            }
        }
        public List<Tensor> TensorsIndependantOfBatchSize
        {
            get { return Layers.SelectMany(x => x.TensorsIndependantOfBatchSize).Where(x => x != null).ToList(); }
        }
        public int TotalParams => Layers.Select(x => x.TotalParams).Sum();
      

        public Tensor NewNotInitializedTensor(int[] shape, Tensor bufferIfAny, string description)
        {
            //we check if we can re use the buffer 'bufferIfAny'
            if (bufferIfAny != null)
            {
                bufferIfAny.AssertIsNotDisposed();
                bufferIfAny.Reshape(shape);
                return bufferIfAny;
            }
            return NewNotInitializedTensor(shape, description);
        }
        public Tensor NewNotInitializedTensor(int[] shape, string description)
        {
            return UseGPU
                ? (Tensor)new GPUTensor<float>(shape, description, GpuWrapper)
                : new CpuTensor<float>(shape, null, description);
        }
        #region serialization
        // ReSharper disable once UnusedMember.Global
        public static Network ValueOf(string path, int?overrideGpuDeviceId = null)
        {
            var content = File.ReadAllLines(path);
            var dicoFirstLine = Serializer.Deserialize(content[0], null);
            var config = NetworkConfig.ValueOf(dicoFirstLine);
            var imageDataGenerator = ImageDataGenerator.ValueOf(dicoFirstLine);
            var gpuDeviceId = overrideGpuDeviceId ?? (int)dicoFirstLine[nameof(_gpuDeviceId)];
            var epochsData = (EpochData[])dicoFirstLine[nameof(_epochsData)];
            var network = new Network(config, imageDataGenerator, gpuDeviceId, epochsData.ToList());
            network.Description = dicoFirstLine.TryGet<string>(nameof(Description))??"";
            for (int i = 1; i < content.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(content[i], network.GpuWrapper), network));
            }
            return network;
        }
        public void Save(string fileName)
        {
            var firstLine = new Serializer()
                .Add(nameof(Description), Description)
                .Add(Config.Serialize())
                .Add(_imageDataGenerator.Serialize())
                .Add(nameof(_gpuDeviceId), _gpuDeviceId)
                .Add(nameof(_epochsData), _epochsData.ToArray())
                .ToString();
            File.AppendAllLines(fileName, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(fileName, new[] { l.Serialize() });
            }
        }
        public void LogContent()
        {
            Layers.ForEach(l => l.LogContent());
        }
        #endregion

        [SuppressMessage("ReSharper", "UnusedParameter.Local")]
        private void CheckInput(IDataSetLoader trainingDateSetCpu, IDataSetLoader testDateSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int miniBatchSize)
        {
            if (trainingDateSetCpu.TypeSize != Config.TypeSize)
            {
                throw new Exception("Invalid type : expecting: "+Config.TypeSize+" but was "+ trainingDateSetCpu.TypeSize);
            }
            foreach (var l in Layers)
            {
                l.CheckConsistency();
            }
            
        }

        public double FindBestLearningRate(IDataSetLoader trainingDataSet, int miniBatchSize = -1)
        {
            Info("Looking for best learning rate...");
            ResetWeights(); //restore weights to there original values
            if (miniBatchSize < 1)
            {
                miniBatchSize = MaxMiniBatchSize();
            }
            else
            {
                if (miniBatchSize > MaxMiniBatchSize())
                {
                    Info("Reducing BatchSize from "+miniBatchSize+" to "+MaxMiniBatchSize());
                    miniBatchSize = MaxMiniBatchSize();
                }
            }
            var learningRateFinder = new LearningRateFinder(miniBatchSize, trainingDataSet.Count);
            bool CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch, int blockIdInEpoch, int nbBatchBlockInEpoch, int epoch)
            {
                bufferComputeLoss = NewNotInitializedTensor(new[] { yExpectedMiniBatch.Shape[0] }, bufferComputeLoss, nameof(bufferComputeLoss));
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, bufferComputeLoss);
                return learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescent(trainingDataSet, miniBatchSize, learningRateFinder, CallBackAfterEachMiniBatch);
            var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_LearningRateFinder.csv");
            File.WriteAllText(fileName, learningRateFinder.AsCsv());
            Info("Stats stored in: " + fileName);
            var bestLearningRate = learningRateFinder.BestLearningRate();
            Info("Best learning rate: "+ bestLearningRate+ " (with batch size="+miniBatchSize+")");
            ResetWeights(); //restore weights to there original values
            return bestLearningRate;
        }


        private string MiniBatchLossFile => Path.Combine(Config.LogDirectory, UniqueId + "_MiniBatchLoss.csv");

        bool CallBackComputeLossAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch, int blockIdInEpoch, int nbBatchBlockInEachEpoch, int epoch)
        {
            var fileName = MiniBatchLossFile;
            if (!File.Exists(fileName))
            {
                File.WriteAllText(fileName, "Sep=;"+Environment.NewLine+"Epoch;Iteration;Loss"+Environment.NewLine);
            }
            _swComputeLoss?.Start();
            bufferComputeLoss = NewNotInitializedTensor(new[] { yExpectedMiniBatch.Shape[0] }, bufferComputeLoss, nameof(bufferComputeLoss));
            var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, bufferComputeLoss);
            _swComputeLoss?.Stop();
            int iteration = (epoch - 1) * nbBatchBlockInEachEpoch + blockIdInEpoch;
            File.AppendAllText(fileName, epoch+";"+ iteration + ";"+blockLoss.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
            return false;
        }


        public void Fit(IDataSetLoader trainingDataSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int preferredMiniBatchSize, IDataSetLoader testDataSetCpuIfAny)
        {
            try
            {
                Debug.Assert(Config.TypeSize == trainingDataSetCpu.TypeSize);
                Debug.Assert(learningRateComputer != null);
                _spInternalFit.Start();
                CheckInput(trainingDataSetCpu, testDataSetCpuIfAny, learningRateComputer, numEpochs, preferredMiniBatchSize);
                
                Info(ToString());
                var maxMiniBatchSize = MaxMiniBatchSize();
                var miniBatchSize = preferredMiniBatchSize;
                if (miniBatchSize < 1)
                {
                    miniBatchSize = maxMiniBatchSize;
                    Info("Using (auto) MiniBatchSize of " + miniBatchSize);
                }
                else if (miniBatchSize > maxMiniBatchSize)
                {
                    Info("Reducing MiniBatchSize from "+ miniBatchSize+" to "+ maxMiniBatchSize+" because of memory limit.");
                    miniBatchSize = maxMiniBatchSize;
                }


                var nbBlocksInEpoch = NbBlocksInEpoch(miniBatchSize, trainingDataSetCpu.Count);
                if (UseGPU)
                {
                    LogDebug(GpuWrapper.ToString());
                }
                LogDebug("Training Set: " + trainingDataSetCpu);
                if (testDataSetCpuIfAny != null)
                {
                    LogDebug("Test Set: " + testDataSetCpuIfAny);
                }
                Info("#Epochs=" + numEpochs + " BathSize=" + miniBatchSize+" Name="+Description);
                if (Config.DisplayTensorContentStats)
                {
                    LogDebug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                }

                Func<Tensor, Tensor, int, int, int, bool> callBackAtEachIteration = null;
                if (Config.SaveLossAfterEachMiniBatch)
                {
                    Info("Saving mini batch loss in " + MiniBatchLossFile);
                    callBackAtEachIteration = CallBackComputeLossAfterEachMiniBatch;
                }

                //Info(GpuWrapper.ToString());

                var lastAutoSaveTime = DateTime.Now; //last time we saved the network
                Tuple<double, double> validationLossAndAccuracy = null;
                for (;;)
                {
                    int epoch = _epochsData.Count + 1;
                    if (epoch > numEpochs)
                    {
                        break;
                    }

                    var swEpoch = Stopwatch.StartNew();

                    var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputer.MultiplicativeFactorFromReduceLrOnPlateau(_epochsData);
                    if (learningRateComputer.ShouldReduceLrOnPlateau(_epochsData))
                    {
                        Info("Reducing learningRate because of plateau at epoch " + epoch + " (new multiplicative coeff:"+ lrMultiplicativeFactorFromReduceLrOnPlateau+")");
                    }

                    #region Mini Batch gradient descent
                    var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, nbBlocksInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var yPredicted = MiniBatchGradientDescent(trainingDataSetCpu, miniBatchSize, learningRateComputer, callBackAtEachIteration);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Config.DisplayTensorContentStats)
                    {
                        LogDebug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine+ContentStats()+Environment.NewLine);
                    }

                    _swComputeLossAndAccuracy?.Start();
                    var trainLossAndAccuracyForEpoch = ComputeLossAndAccuracyForEntireBatch(_yExpectedBufferForEntireBatch, yPredicted);
                    var lossAndAccuracyMsg = LossAndAccuracyToString(trainLossAndAccuracyForEpoch, "");
                    if (testDataSetCpuIfAny != null)
                    {
                        //We compute the validation (= test) loss&accuracy
                        validationLossAndAccuracy = ComputeLossAndAccuracyForTestDataSet(miniBatchSize, testDataSetCpuIfAny);
                        lossAndAccuracyMsg += " - "+LossAndAccuracyToString(validationLossAndAccuracy, "val_");
                    }
                    _swComputeLossAndAccuracy?.Stop();
                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataSetCpu.Count) / miniBatchSize;
                    var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                    Info("Epoch " + epoch + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRateAtEpochStart, 8)+" - "+lossAndAccuracyMsg);
                    if (UseGPU)
                    {
                        Info(GpuWrapper.MemoryInfo());
                    }
                    if (Config.ProfileApplication)
                    {
                        LogDebug(ProfilingComments());
                    }

                    #region we save stats about the just finished epoch
                    var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, trainLossAndAccuracyForEpoch.Item1, trainLossAndAccuracyForEpoch.Item2, validationLossAndAccuracy?.Item1 ?? double.NaN, validationLossAndAccuracy?.Item2 ?? double.NaN, secondsForEpoch);
                    _epochsData.Add(currentEpochData);
                    #endregion

                    #region we save the network in a file if necessary
                    if (   //if we have finished training
                           ((epoch == numEpochs) && (numEpochs > 10))
                            //or if we should save the network every 'Config.AutoSaveIntervalInMinuts' minuts
                        || ( (Config.AutoSaveIntervalInMinutes>=0) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinutes)
                        || learningRateComputer.ShouldCreateSnapshotForEpoch(epoch)
                        )
                    {
                        var swSaveTime = Stopwatch.StartNew();
                        var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + ".txt");
                        Save(fileName);
                        Info("Network '" + Description + "' saved in " + fileName + " in " + Math.Round(swSaveTime.Elapsed.TotalSeconds, 1) + "s");
                        lastAutoSaveTime = DateTime.Now;
                    }
                    #endregion

                    if (Config.SaveNetworkStatsAfterEachEpoch)
                    {
                        var networkStatFileName = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + "_NetworkStats.txt");
                        Info("Saving network '" + Description + "' stats in " + networkStatFileName);
                        File.WriteAllText(networkStatFileName, ContentStats());
                    }
                }

                string line = "";
                try
                {
                    //We save the results of the net
                    line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                        + Description.Replace(';', '_') + ";"
                        + DeviceName() + ";"
                        + TotalParams + ";"
                        + numEpochs + ";"
                        + miniBatchSize + ";"
                        + learningRateComputer.LearningRate(1, 0, nbBlocksInEpoch, 1.0) + ";"
                        + _spInternalFit.Elapsed.TotalSeconds + ";"
                        + (_spInternalFit.Elapsed.TotalSeconds / numEpochs) + ";"
                        + validationLossAndAccuracy?.Item1 + ";"
                        + validationLossAndAccuracy?.Item2
                        + Environment.NewLine;
                    if (!Config.DisableLogging)
                    {
                        var testsCsv = string.IsNullOrEmpty(trainingDataSetCpu.Name)?"Tests.csv": ("Tests_"+ trainingDataSetCpu.Name + ".csv");
                        File.AppendAllText(Utils.ConcatenatePathWithFileName(Config.LogDirectory, testsCsv), line);
                    }
                }
                catch (Exception e)
                {
                    Info("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
                    // ignored
                }

                Info("Training '"+ Description+"' for " + numEpochs + " epochs took: " + _spInternalFit.Elapsed.TotalSeconds + "s");
                if (!string.IsNullOrEmpty(Description))
                {
                    LogDebug("Network Name: "+Description);
                }
                _spInternalFit.Stop();
            }
            catch (Exception e)
            {
                Info(e.ToString());
                throw;
            }
        }

        private string ContentStats()
        {
            var sb = new StringBuilder();
            foreach (var l in Layers)
            {
                sb.Append(new string('-',80)+Environment.NewLine);
                sb.Append("Layer:" + l.SummaryName() + Environment.NewLine);
                var contentStats = l.ContentStats();
                if (!string.IsNullOrEmpty(contentStats))
                {
                    sb.Append(contentStats + Environment.NewLine);
                }
            }
            return sb.ToString();
        }

        #region compute Loss and Accuracy
        //returns : Tuple<loss, accuracy>
        public Tuple<double, double> ComputeLossAndAccuracyForTestDataSet(int miniBatchSize, IDataSetLoader testDataSet)
        {
            //We perform a mini batch gradient descent in Testing mode:
            //  there will be no shuffling/data augmentation.
            var yPredicted = MiniBatchGradientDescent(testDataSet, miniBatchSize, null, null);
            return ComputeLossAndAccuracyForEntireBatch(testDataSet.Y, yPredicted);
        }

        private Tuple<double, double> ComputeLossAndAccuracyForEntireBatch(Tensor yExpected, Tensor yPredicted)
        {
            _swComputeAccuracy?.Start();
            yExpected = ReformatToCorrectDevice_GPU_or_CPU(yExpected);
            yPredicted = ReformatToCorrectDevice_GPU_or_CPU(yPredicted);
            bufferComputeAccuracy = NewNotInitializedTensor(new []{ yExpected.Shape[0]}, bufferComputeAccuracy, nameof(bufferComputeAccuracy));
            var accuracy = yExpected.ComputeAccuracy(yPredicted, bufferComputeAccuracy);
            _swComputeAccuracy?.Stop();
            _swComputeLoss?.Start();
            bufferComputeLoss = NewNotInitializedTensor(new[] { yExpected.Shape[0] }, bufferComputeLoss, nameof(bufferComputeLoss));
            var totalLoss = yExpected.ComputeLoss(yPredicted, Config.LossFunction, bufferComputeLoss);
            _swComputeLoss?.Stop();
            return Tuple.Create(totalLoss, accuracy);
        }
        private static string LossAndAccuracyToString(Tuple<double, double> lossAndAccuracy, string prefix)
        {
            return prefix + "loss: " + Math.Round(lossAndAccuracy.Item1, 4) + " - " + prefix + "acc: " + Math.Round(lossAndAccuracy.Item2, 4);
        }
        #endregion

        //private ulong OccupiedMemoryInBytes => _layers.Select(x => x.OccupiedMemoryInBytes).Sum();
        private ulong BytesByBatchSize
        {
            get
            {
                return Layers.Select(x => x.BytesByBatchSize).Sum()+_backwardPropagationManager.BytesByBatchSizeForGradientComputation;
            }
        }

        private ulong BytesIndependantOfBatchSize => Layers.Select(x => x.BytesIndependantOfBatchSize).Sum();

        private Tensor ReformatToCorrectDevice_GPU_or_CPU(Tensor X)
        {
            if (X == null || UseGPU == X.UseGPU)
            {
                return X;
            }
            if (X.UseGPU)
            {
                throw new NotImplementedException("can not reformat type that are stored in GPU");
            }
            return X.ToGPU<float>(GpuWrapper);
        }
        private string ProfilingComments()
        {
            var totalMs = (double)_spInternalFit.ElapsedMilliseconds;
            var result = "Took " + Math.Round(totalMs / 1000.0, 1) + "s";
            if (Config.ProfileApplication)
            {
                result += " (";
                result += "ForwardPropagation [Training:" + Math.Round(100 * _swPredictTraining.ElapsedMilliseconds / totalMs, 0) + "% / not Training:" + Math.Round(100 * _swPredictNotTraining.ElapsedMilliseconds / totalMs, 0) + "%] ";
                result += ", BackwardPropagation:" + Math.Round(100 * _swBackwardPropagation.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", UpdateWeights:" + Math.Round(100 * _swUpdateWeights.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", CreateInputForEpoch:" + Math.Round(100 * _swCreateInputForEpoch.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", ComputeLossAndAccuracy:" + Math.Round(100 * _swComputeLossAndAccuracy.ElapsedMilliseconds / totalMs, 0) + "%";
                result += " [Loss:" + Math.Round(100 * _swComputeLoss.ElapsedMilliseconds / totalMs, 0) + "%+Accuracy:"+ Math.Round(100 * _swComputeAccuracy.ElapsedMilliseconds / totalMs, 0) +"%]";
                result += ")";
            }
            return result;
        }
        private void UpdateWeights(double learningRate)
        {
            _swUpdateWeights?.Start();
            foreach (var l in Layers.Skip(1)) //we skip the input layer
            {
                l.UpdateWeights(learningRate);
            }
            _swUpdateWeights?.Stop();
        }
        public void Info(string msg) { Config.Logger.Info(msg); }
        public void LogDebug(string msg) { Config.Logger.Debug(msg); }

        /// <summary>
        /// Perform a mini batch gradient descent for an entire epoch, each mini batch will have 'miniBatchSize' elements
        /// </summary>
        /// <param name="dataSet">Expected Input and output (= dataSet.Y) </param>
        /// <param name="miniBatchSize"></param>
        /// <param name="learningRateComputerIfTraining">null if we are just using the network to predict the results (without updating weights)
        ///     not null if we need to update the weights between each mini batch</param>
        /// <param name="callBackToStop">Optional callback to be called at the end of each mini batch,
        ///     parameters are: 'mini batch expected output' + 'mini batch observed output' + 'current block Id'
        ///     If the callback returns true we should stop the computation</param>
        /// <returns>observed output associated with the input 'x'</returns>
        public Tensor MiniBatchGradientDescent(IDataSetLoader dataSet, int miniBatchSize = -1,
            ILearningRateComputer learningRateComputerIfTraining = null,
            Func<Tensor, Tensor, int, int, int, bool> callBackToStop = null)
        {
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering MiniBatchGradientDescent(" + miniBatchSize + "); epoch="+ (_epochsData.Count + 1));}
#endif
            bool isTraining = learningRateComputerIfTraining != null;
            var entireBatchSize = dataSet.Count;
            if (miniBatchSize <= 0)
            {
                miniBatchSize = MaxMiniBatchSize();
            }
            int nbMiniBatchBlock = NbBlocksInEpoch(miniBatchSize, entireBatchSize);
            int epoch = _epochsData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(_epochsData) ?? 1.0;
            _yPredictedBufferForEntireBatch = NewNotInitializedTensor(dataSet.Y_Shape, _yPredictedBufferForEntireBatch, nameof(_yPredictedBufferForEntireBatch));
            _yExpectedBufferForEntireBatch = NewNotInitializedTensor(dataSet.Y_Shape, _yExpectedBufferForEntireBatch, nameof(_yExpectedBufferForEntireBatch));
            var xMiniBatch = NewNotInitializedTensor(dataSet.XMiniBatch_Shape(miniBatchSize), "xMiniBatch");

            var orderInCurrentEpoch = Enumerable.Range(0, dataSet.Count).ToList();
            if (epoch >= 2 && Config.RandomizeOrder && isTraining) 
            {
                Utils.Shuffle(orderInCurrentEpoch, Config.Rand);
            }

            int blockId = 0;
            int nbProcessed = 0;
            while(nbProcessed < entireBatchSize)
            {
#if DEBUG
                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("start MiniBatchGradientDescent(" + miniBatchSize + "); epoch=" + (_epochsData.Count + 1)+"; blockId="+blockId);}
#endif
                var blockSize = Math.Min(entireBatchSize- nbProcessed, miniBatchSize);
                xMiniBatch.Reshape(dataSet.XMiniBatch_Shape(blockSize));

                var yExpectedMiniBatch = _yExpectedBufferForEntireBatch.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                _swCreateInputForEpoch?.Start();
                dataSet.Load(epoch, isTraining, blockId * miniBatchSize, orderInCurrentEpoch, _imageDataGenerator, ref xMiniBatch, ref yExpectedMiniBatch);
                _swCreateInputForEpoch?.Stop();

                var yPredictedMiniBatch = _yPredictedBufferForEntireBatch.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                Layers.Last().Set_y(yPredictedMiniBatch);
                Predict(xMiniBatch, isTraining);
                if (isTraining)
                {
                    BackwardPropagation(yExpectedMiniBatch);
                    UpdateWeights(learningRateComputerIfTraining.LearningRate(epoch, blockId, nbMiniBatchBlock, lrMultiplicativeFactorFromReduceLrOnPlateau));
                }
                if (!yPredictedMiniBatch.UseGPU)
                {
                    yPredictedMiniBatch.CopyTo(0, _yPredictedBufferForEntireBatch, _yPredictedBufferForEntireBatch.Idx(nbProcessed), yPredictedMiniBatch.Count);
                    yExpectedMiniBatch.CopyTo(0, _yExpectedBufferForEntireBatch, _yExpectedBufferForEntireBatch.Idx(nbProcessed),  yExpectedMiniBatch.Count);
                }
                nbProcessed += blockSize;
                if (callBackToStop != null && callBackToStop(yExpectedMiniBatch, yPredictedMiniBatch, blockId, nbMiniBatchBlock, epoch))
                {
                    break;
                }
#if DEBUG
                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("end MiniBatchGradientDescent(" + miniBatchSize + "); epoch=" + (_epochsData.Count + 1) + "; blockId=" + blockId);}
#endif
                ++blockId;
            }
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving MiniBatchGradientDescent(" + miniBatchSize + "); epoch=" + (_epochsData.Count + 1));}
#endif
            return _yPredictedBufferForEntireBatch;
        }
        private static int NbBlocksInEpoch(int miniBatchSize, int entireBatchSize)
        {
            return (entireBatchSize + miniBatchSize - 1) / miniBatchSize;
        }
    }
}
