using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNet
{
    public class Network
    {
        #region fields
        private int _indexCurrentEpochs;
        private readonly ImageDataGenerator _imageDataGenerator;
        public NetworkConfig Config { get; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { get; set; } = "";
        private readonly Stopwatch spInternalFit = new Stopwatch();
        private readonly Stopwatch swUpdateWeights;
        private readonly Stopwatch swPredictTraining;
        private readonly Stopwatch swPredictNotTraining;
        private readonly Stopwatch swBackwardPropagation;
        private readonly Stopwatch swCreateInputForEpoch;
        private readonly Stopwatch swComputeLossAndAccuracy;
        private readonly Stopwatch swComputeLoss;
        private readonly Stopwatch swComputeAccuracy;
        private Tensor _yPredictedBufferForMiniBatchGradientDescent;
        #endregion

        public Network(NetworkConfig config, ImageDataGenerator imageDataGenerator = null, int indexCurrentEpochs = 1)
        {
            Config = config;
            _imageDataGenerator = imageDataGenerator??ImageDataGenerator.NoDataAugmentation;
            _indexCurrentEpochs = indexCurrentEpochs;

            if (config.ProfileApplication)
            {
                swUpdateWeights = new Stopwatch();
                swPredictTraining = new Stopwatch();
                swPredictNotTraining = new Stopwatch();
                swBackwardPropagation = new Stopwatch();
                swCreateInputForEpoch = new Stopwatch();
                swComputeLossAndAccuracy = new Stopwatch();
                swComputeLoss = new Stopwatch();
                swComputeAccuracy = new Stopwatch();
            }
        }

        #region network construction: adding layers
        public Network AddInput(int channelCount, int h, int w)
        {
            ClearMemory();
            Layers.Add(new InputLayer(channelCount, h, w, this));
            return this;
        }

        public void ClearMemory()
        {
            Config.GpuWrapper?.ClearMemory();
            Layers.ForEach(x => x?.Dispose());
            Layers.Clear();
        }
        public Network AddDense(int n_x, double _lambdaL2Regularization)
        {
            Debug.Assert(Layers.Count >= 1);
            var fullyConnectedLayer = new DenseLayer(n_x, _lambdaL2Regularization, this);
            Layers.Add(fullyConnectedLayer);
            return this;
        }
        public Network AddConvolution_BatchNorm(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return AddConvolution(filtersCount, f, stride, padding, lambdaL2Regularization)
                .AddBatchNorm();
        }
        public Network AddConvolution_BatchNorm_Activation(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return AddConvolution_BatchNorm(filtersCount, f, stride, padding, lambdaL2Regularization)
                .AddActivation(activationFunction);
        }
        public Network AddSumLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex)
        {
            Layers.Add(new SumLayer(previousIdentityLayerIndex, previousResidualLayerIndex, this));
            Debug.Assert(Layers[previousIdentityLayerIndex].SameOutputShape(Layers[previousResidualLayerIndex]));
            return this;
        }
        //add a shortcut from layer 'AddSumLayer' to current layer, adding a Conv Layer if necessary (for matching size)
        public Network AddShortcut_IdentityConnection(int startOfBlockLayerIndex, int filtersCount, int stride, double lambdaL2Regularization)
        {
            int previousResidualLayerIndex = Layers.Last().LayerIndex;

            var sameInputAndOutputShapeInBlock = Layers.Last().SameOutputShape(Layers[startOfBlockLayerIndex]);
            if (sameInputAndOutputShapeInBlock)
            {
                Layers.Add(new SumLayer(startOfBlockLayerIndex, previousResidualLayerIndex, this));
            }
            else
            {
                //we need to add a convolution layer to make correct output format
                AddConvolution(filtersCount, 1, stride, 0, lambdaL2Regularization, startOfBlockLayerIndex);
                int convLayerIdInIdentityBlock = Layers.Last().LayerIndex;
                Layers.Add(new SumLayer(convLayerIdInIdentityBlock, previousResidualLayerIndex, this));
                Debug.Assert(Layers[convLayerIdInIdentityBlock].SameOutputShape(Layers[previousResidualLayerIndex]));
            }
            return this;
        }
        public Network AddConvolution_Activation_Pooling(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, cudnnActivationMode_t activationFunction, int poolingSize, int poolingStride)
        {
            return AddConvolution(filtersCount, f, stride, padding, lambdaL2Regularization)
                .AddActivation(activationFunction)
                .AddMaxPooling(poolingSize, poolingStride);
        }
        public Network AddConvolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return AddConvolution(filtersCount, f, stride, padding, lambdaL2Regularization, Layers.Count - 1);
        }
        public Network AddConvolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, int previousLayerIndex)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(filtersCount, f, stride, padding, lambdaL2Regularization, previousLayerIndex, this));
            return this;
        }
        public Network AddDropout(double dropProbability)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new DropoutLayer(dropProbability, this));
            return this;
        }
        public Network AddActivation(cudnnActivationMode_t activationFunction)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ActivationLayer(activationFunction, this));
            return this;
        }
        public Network AddMaxPooling(int poolingSize, int poolingStride)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingStride,
                this));
            return this;
        }
        public Network AddAvgPooling(int poolingSize, int poolingStride)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, poolingSize,
                poolingStride, this));
            return this;
        }
        public Network AddGlobalAvgPooling()
        {
            var lastLayerShape = Layers.Last().OutputShape(1);
            var lastLayerShapeHeight = lastLayerShape[2];
            //We ensure that weight and height are the same
            Debug.Assert(lastLayerShapeHeight == lastLayerShape[3]);
            int poolingSize = lastLayerShapeHeight;
            int poolingStride = lastLayerShapeHeight;
            return AddAvgPooling(poolingSize, poolingStride);
        }
        public Network AddBatchNorm(double momentum = 0.99, double epsilon = 1e-5)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new BatchNormalizationLayer(momentum, epsilon, this));
            return this;
        }
        public Network AddDense_Activation(int n_x, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return AddDense(n_x, lambdaL2Regularization)
                .AddActivation(activationFunction);
        }
        public Network AddOutput(int n_x, double _lambdaL2Regularization, cudnnActivationMode_t activationFunctionType)
        {
            return AddDense(n_x, _lambdaL2Regularization)
                .AddActivation(activationFunctionType);
        }
        public Network Flatten()
        {
            Debug.Assert(Layers.Count >= 1);
            var flattenLayer = new FlattenLayer(this);
            Layers.Add(flattenLayer);
            return this;
        }
        #endregion

        public void Fit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, double learningRate, int numEpochs, int batchSize, CpuTensor<T> X_test = null, CpuTensor<T> Y_test = null) where T : struct
        {
            Fit(xCpu, yCpu, LearningRateScheduler.Constant(learningRate), numEpochs, batchSize, X_test, Y_test);
        }
        public void Fit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, LearningRateScheduler learningRateScheduler, int numEpochs, int batchSize, CpuTensor<T> X_testCpu = null, CpuTensor<T> Y_testCpu = null) where T : struct
        {
            if (Config.UseDoublePrecision)
            {
                InternalFit(xCpu.ToDoublePrecision(), yCpu.ToDoublePrecision(), learningRateScheduler, numEpochs, batchSize, X_testCpu.ToDoublePrecision(), Y_testCpu.ToDoublePrecision());
            }
            else
            {
                InternalFit(xCpu.ToSinglePrecision(), yCpu.ToSinglePrecision(), learningRateScheduler, numEpochs, batchSize, X_testCpu.ToSinglePrecision(), Y_testCpu.ToSinglePrecision());
            }
        }
        //= ForwardPropagation
        public Tensor Predict(Tensor X, bool isTraining)
        {
            (isTraining ? swPredictTraining : swPredictNotTraining)?.Start();
            X = ReformatToCorrectType(X);
            ((InputLayer)Layers[0]).Set_y(X);
            foreach (var l in Layers.Skip(1))
            {
                l.ForwardPropagation(isTraining);
                //Info(Environment.NewLine+"Epoch:" + _indexCurrentEpochs + "; Layer:" + l.SummaryName() + "_" + l.LayerIndex + "; After ForwardPropagation:" + Environment.NewLine + l.ContentStats());
            }
            (isTraining ? swPredictTraining : swPredictNotTraining)?.Stop();
            return Layers.Last().y;
        }
        public void BackwardPropagation(Tensor yExpected)
        {
            swBackwardPropagation?.Start();

            var yPredicted = Layers.Last().y;
            Debug.Assert(yPredicted != null);
            Debug.Assert(yExpected.SameShape(yPredicted));

            //we compute: dyPredicted = 0.5*(yPredicted - yExpected)
            //!D TODO : do the 2 following steps once
            var dyPredicted = Layers.Last().dy;
            yPredicted.CopyTo(dyPredicted);
            dyPredicted.Update_Adding_Alpha_X(-1.0, yExpected);

            if (Layers.Last().IsSigmoidActivationLayer())
            {
                var categoryCount = yPredicted.Shape[1];
                dyPredicted.Update_Multiplying_By_Alpha(1.0 / categoryCount);
            }

            for (int i = Layers.Count - 1; i >= 1; --i)
            {
                Layers[i].BackwardPropagation();
                //Info(Environment.NewLine + "Epoch:" + _indexCurrentEpochs + "; Layer:" + Layers[i].SummaryName() + "_" + Layers[i].LayerIndex + "; After BackwardPropagation:" + Environment.NewLine + Layers[i].ContentStats());
            }
            swBackwardPropagation?.Stop();
        }
        public string Summary()
        {
            const string line0 = "_________________________________________________________________";
            const string line1 = "=================================================================";
            string result = "";
            if (!string.IsNullOrEmpty(Description))
            {
                result += "Network Name: " + Description+ Environment.NewLine;
            }
            result += line0 + Environment.NewLine;
            result += "Layer (C#)                   Output Shape              Param #" + Environment.NewLine;
            result += line1 + Environment.NewLine;
            foreach (var l in Layers)
            {
                var outputShape = Utils.ShapeToStringWithBacthSize(l.OutputShape(1));
                result += $"{l.SummaryName(),-29}{outputShape,-26}{l.TotalParams}" + Environment.NewLine;
                result += (l.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams;
            return result;
        }
        public override string ToString()
        {
            var result = Summary() + Environment.NewLine;
            result += Utils.MemoryBytesToString(BytesByBatchSize) + "/batchSize+" + Utils.MemoryBytesToString(BytesIndependantOfBatchSize);
            return result;
        }
        public Optimizer GetOptimizer(int[] weightShape, int[] biasShape)
        {
            switch (Config.OptimizerType)
            {
                case Optimizer.OptimizationEnum.Adam: return new Adam(this, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD: return new Sgd(this, weightShape, biasShape);
                default: return VanillaSgd.Instance;
            }
        }
        public List<Tensor> TensorsIndependantOfBatchSize
        {
            get { return Layers.SelectMany(x => x.TensorsIndependantOfBatchSize).Where(x => x != null).ToList(); }
        }
        public Tensor RandomMatrixNormalDistribution(int[] shape, double mean, double stdDev, string description)
        {
            if (Config.UseDoublePrecision)
            {
                var doubleCpuTensor = CpuTensor<double>.RandomDoubleNormalDistribution(shape, Config.Rand, mean, stdDev, description);
                return Config.UseGPU ? (Tensor)doubleCpuTensor.ToGPU<double>(Config.GpuWrapper) : doubleCpuTensor;
            }
            var floatCpuTensor = CpuTensor<float>.RandomFloatNormalDistribution(shape, Config.Rand, mean, stdDev, description);
            return Config.UseGPU ? (Tensor)floatCpuTensor.ToGPU<float>(Config.GpuWrapper) : floatCpuTensor;
        }
        public int TotalParams => Layers.Select(x => x.TotalParams).Sum();
        public Tensor NewTensor(int[] shape, Tensor bufferIfAny, string description)
        {
            //we check if we can re use the buffer 'bufferIfAny'
            if (bufferIfAny != null && bufferIfAny.CapacityInBytes >= bufferIfAny.ReallyNeededMemoryInBytesForShape(shape))
            {
                bufferIfAny.Reshape(shape);
                return bufferIfAny;
            }
            return NewTensor(shape, description);
        }
        public Tensor NewTensor(int[] shape, string description)
        {
            if (Config.UseDoublePrecision)
            {
                return Config.UseGPU
                    ? (Tensor)new GPUTensor<double>(shape, null, description, Config.GpuWrapper)
                    : new CpuTensor<double>(shape, null, description);
            }
            else
            {
                return Config.UseGPU
                    ? (Tensor)new GPUTensor<float>(shape, null, description, Config.GpuWrapper)
                    : new CpuTensor<float>(shape, null, description);
            }
        }
        public Tensor NewTensor(int[] shape, double sameValue, string description)
        {
            return Config.UseDoublePrecision
                ? NewTensorOfSameValue(shape, sameValue, description)
                : NewTensorOfSameValue(shape, (float)sameValue, description);
        }

        #region serialization
        public static Network ValueOf(string path)
        {
            var content = File.ReadAllLines(path);
            var dicoFirstLine = Serializer.Deserialize(content[0], null);
            var config = NetworkConfig.ValueOf(dicoFirstLine);
            var imageDataGenerator = ImageDataGenerator.ValueOf(dicoFirstLine);
            var indexCurrentEpochs = (int)dicoFirstLine[nameof(_indexCurrentEpochs)];
            var network = new Network(config, imageDataGenerator, indexCurrentEpochs);
            for (int i = 1; i < content.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(content[i], network.Config.GpuWrapper), network));
            }
            return network;
        }
        private void Save(string path)
        {
            var fileName = Path.Combine(path, "Network_" + Process.GetCurrentProcess().Id + "_" + _indexCurrentEpochs + ".txt");
            var firstLine = new Serializer()
                .Add(Config.Serialize())
                .Add(_imageDataGenerator.Serialize())
                .Add(nameof(_indexCurrentEpochs), _indexCurrentEpochs + 1)
                .ToString();
            File.AppendAllLines(fileName, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(fileName, new[] { l.Serialize() });
            }
        }
        public void LogContent()
        {
            for (int i = 0; i < Layers.Count; ++i)
            {
                Layers[i].LogContent();
            }
        }
        #endregion

        [SuppressMessage("ReSharper", "UnusedParameter.Local")]
        private void CheckInput<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, LearningRateScheduler learningRateScheduler, int numEpochs, int batchSize, CpuTensor<T> X_testCpu, CpuTensor<T> Y_testCpu) where T : struct
        {
            Debug.Assert(xCpu.Shape[0] == yCpu.Shape[0]); //same number of tests
            if (!Layer.IsValidYSet(yCpu))
            {
                throw new Exception("Tnvalid Training Set 'y' : must contain only 0 and 1");
            }
            if (X_testCpu != null && Y_testCpu != null)
            {
                Debug.Assert(X_testCpu.Shape[0] == Y_testCpu.Shape[0]); //same number of tests
                Debug.Assert(yCpu.Shape[1] == Y_testCpu.Shape[1]);
                if (!Layer.IsValidYSet(yCpu))
                {
                    throw new Exception("Tnvalid Test Set 'Y_test' : must contain only 0 and 1");
                }
            }
            var observedTypeSize = Marshal.SizeOf(typeof(T));
            if (observedTypeSize != Config.TypeSize)
            {
                throw new Exception("Tnvalid type : expecting: "+Config.TypeSize+" but was "+ observedTypeSize);
            }
            foreach (var l in Layers)
            {
                l.CheckConsistency();
            }
        }
        //here T is already of the target precision (double or float)
        private void InternalFit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, LearningRateScheduler learningRateScheduler, int numEpochs, int batchSize, CpuTensor<T> xTestCpu, CpuTensor<T> yTestCpu) where T : struct
        {
            Debug.Assert(Config.TypeSize == xCpu.TypeSize);
            Debug.Assert(Config.TypeSize == yCpu.TypeSize);
            spInternalFit.Restart();

            CheckInput(xCpu, yCpu, learningRateScheduler, numEpochs, batchSize, xTestCpu, yTestCpu);
            var yInputCpu = yCpu.Clone();
            var X = ReformatToCorrectDevice_GPU_or_CPU(xCpu);
            var Y = ReformatToCorrectDevice_GPU_or_CPU(yCpu);
            var xTest = ReformatToCorrectDevice_GPU_or_CPU(xTestCpu);
            var yTest = ReformatToCorrectDevice_GPU_or_CPU(yTestCpu);
            Info(ToString());
            if (Config.UseGPU)
            {
                LogDebug(Config.GpuWrapper.ToString());
            }

            var enlargedXCpu = _imageDataGenerator.EnlargePictures(xCpu);
            var lastAutoSaveTime = DateTime.Now; //last time we saved the network
            for (;_indexCurrentEpochs <= numEpochs; ++_indexCurrentEpochs)
            {
                var swEpoch = Stopwatch.StartNew();
                double learningRate = learningRateScheduler.LearningRate(_indexCurrentEpochs);
                if (_indexCurrentEpochs != 1) //for the very fist epoch we use exactly the same input, with no shuffling or data augmentation
                {
                    swCreateInputForEpoch?.Start();
                    _imageDataGenerator.CreateInputForEpoch(enlargedXCpu, yInputCpu, xCpu, yCpu, Config.RandomizeOrder);
                    swCreateInputForEpoch?.Stop();
                }
                if (X.UseGPU)
                {
                    ((GPUTensor<T>)X).CopyToDevice();
                    ((GPUTensor<T>)Y).CopyToDevice();
                }

                if (_indexCurrentEpochs == 1)
                {
                    LogDebug("Training Set: " + X + " => " + Y);
                    if (xTest != null)
                    {
                        LogDebug("Test Set: " + xTest + " => " + yTest);
                    }
                    Info("LearningRate=" + learningRate + " #Epochs=" + numEpochs + " BathSize=" + batchSize);
                }
                var yPredicted = MiniBatchGradientDescent(batchSize, true, learningRate, X, Y);

                //We display stats about the just finished epoch
                swComputeLossAndAccuracy?.Start();
                var computeLossAndAccuracy = LossAndAccuracyToString(ComputeLossAndAccuracy_From_Expected_vs_Predicted(Y, yPredicted, Config.LossFunction), "");
                if (xTest != null)
                {
                    computeLossAndAccuracy += " - "+LossAndAccuracyToString(ComputeLossAndAccuracy(batchSize, xTest, yTest), "val_");
                }
                swComputeLossAndAccuracy?.Stop();
                var secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                double nbStepsByEpoch = ((double)X.Shape[0]) / batchSize;
                var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                Info("Epoch " + _indexCurrentEpochs + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRate, 5)+" - "+computeLossAndAccuracy);
                if (Config.UseGPU)
                {
                    LogDebug(Config.GpuWrapper.MemoryInfo());
                }
                if (Config.ProfileApplication)
                {
                    LogDebug(ProfilingComments());
                }

                if (  ((_indexCurrentEpochs == numEpochs)&&(numEpochs>=10))
                    || (!string.IsNullOrEmpty(Config.AutoSavePath) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinuts) )
                {
                    Info("Saving network in directory '"+Config.AutoSavePath+"' ...");
                    var swSaveTime = Stopwatch.StartNew();
                    Save(Config.AutoSavePath);
                    Info("Network saved in directory '" + Config.AutoSavePath + "' in "+ Math.Round(swSaveTime.Elapsed.TotalSeconds,1)+ "s");
                    lastAutoSaveTime = DateTime.Now;
                }
            }
            Info("Training for " + numEpochs + " epochs took: " + spInternalFit.Elapsed.TotalSeconds + "s");
            if (!string.IsNullOrEmpty(Description))
            {
                LogDebug("Network Name: "+Description);
            }
            _indexCurrentEpochs = 1;
            if (X.UseGPU)
            {
                X.Dispose();
            }
        }

        #region compute Loss and Accuracy
        //returns : Tuple<loss, accuracy>
        public Tuple<double, double> ComputeLossAndAccuracy(int miniBatchSize, Tensor X, Tensor yExpected)
        {
            var yPredicted = MiniBatchGradientDescent(miniBatchSize, false, 0.0, X, yExpected);
            return ComputeLossAndAccuracy_From_Expected_vs_Predicted(yExpected, yPredicted, Config.LossFunction);
        }
        private Tuple<double, double> ComputeLossAndAccuracy_From_Expected_vs_Predicted(Tensor yExpected, Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction)
        {
            swComputeAccuracy?.Start();
            yExpected = ReformatToCorrectType(yExpected);
            yPredicted = ReformatToCorrectType(yPredicted);
            var countOk = yExpected.ComputeAccuracy(yPredicted);
            swComputeAccuracy?.Stop();
            swComputeLoss?.Start();
            var totalLoss = yExpected.ComputeLoss(yPredicted, lossFunction);
            swComputeLoss?.Stop();
            return Tuple.Create(totalLoss, countOk / ((double)yExpected.Shape[0]));
        }
        private static string LossAndAccuracyToString(Tuple<double, double> lossAndAccuracy, string prefix)
        {
            return prefix + "loss: " + Math.Round(lossAndAccuracy.Item1, 4) + " - " + prefix + "acc: " + Math.Round(lossAndAccuracy.Item2, 4);
        }
        #endregion

        private Tensor ReformatToCorrectType(Tensor X)
        {
            X = ReformatToCorrectPrecision_float_or_double(X);
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            return X;
        }
        //private ulong OccupiedMemoryInBytes => _layers.Select(x => x.OccupiedMemoryInBytes).Sum();
        private ulong BytesByBatchSize => Layers.Select(x => x.BytesByBatchSize).Sum();
        private ulong BytesIndependantOfBatchSize => Layers.Select(x => x.BytesIndependantOfBatchSize).Sum();
        private Tensor ReformatToCorrectPrecision_float_or_double(Tensor X)
        {
            if (X == null || Config.UseDoublePrecision == X.UseDoublePrecision)
            {
                return X;
            }
            return X.UseDoublePrecision
                ? X.AsDoubleCpu.ToSinglePrecision()
                : (Tensor)X.AsFloatCpu.ToDoublePrecision();
        }
        private Tensor ReformatToCorrectDevice_GPU_or_CPU(Tensor X)
        {
            if (X == null || Config.UseGPU == X.UseGPU)
            {
                return X;
            }
            if (X.UseGPU)
            {
                throw new NotImplementedException("can not reformat type that are stored in GPU");
            }
            return X.UseDoublePrecision
                ? (Tensor)X.ToGPU<double>(Config.GpuWrapper)
                : X.ToGPU<float>(Config.GpuWrapper);
        }
        private string ProfilingComments()
        {
            var totalMs = (double)spInternalFit.ElapsedMilliseconds;
            var result = "Took " + Math.Round(totalMs / 1000.0, 1) + "s";
            if (Config.ProfileApplication)
            {
                result += " (";
                result += "ForwardPropagation [Training:" + Math.Round(100 * swPredictTraining.ElapsedMilliseconds / totalMs, 0) + "% / not Training:" + Math.Round(100 * swPredictNotTraining.ElapsedMilliseconds / totalMs, 0) + "%] ";
                result += ", BackwardPropagation:" + Math.Round(100 * swBackwardPropagation.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", UpdateWeights:" + Math.Round(100 * swUpdateWeights.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", CreateInputForEpoch:" + Math.Round(100 * swCreateInputForEpoch.ElapsedMilliseconds / totalMs, 0) + "%";
                result += ", ComputeLossAndAccuracy:" + Math.Round(100 * swComputeLossAndAccuracy.ElapsedMilliseconds / totalMs, 0) + "%";
                result += " [Loss:" + Math.Round(100 * swComputeLoss.ElapsedMilliseconds / totalMs, 0) + "%+Accuracy:"+ Math.Round(100 * swComputeAccuracy.ElapsedMilliseconds / totalMs, 0) +"%]";
                result += ")";
            }
            return result;
        }
        private void UpdateWeights(double learningRate)
        {
            swUpdateWeights?.Start();
            foreach (var l in Layers.Skip(1)) //we skip the input layer
            {
                l.UpdateWeights(learningRate);
                //Info(Environment.NewLine + "Epoch:" + _indexCurrentEpochs + "; Layer:" + l.SummaryName() + "_" + l.LayerIndex + "; After UpdateWeights:" + Environment.NewLine+l.ContentStats());
            }
            swUpdateWeights?.Stop();
        }
        private void Info(string msg) { Config.Logger.Info(msg); }
        private void LogDebug(string msg) { Config.Logger.Debug(msg); }
        private Tensor MiniBatchGradientDescent(int miniBatchSize, bool isTraining, double learningRateIfTraining, Tensor X, Tensor yExpected)
        {
            X = ReformatToCorrectType(X);
            yExpected = ReformatToCorrectType(yExpected);
            var entireBatchSize = X.Shape[0];
            Debug.Assert(entireBatchSize == yExpected.Shape[0]);
            if (miniBatchSize <= 0)
            {
                throw new Exception("invalid miniBatchSize size (" + miniBatchSize + ")");
            }
            int nbBatchBlock = (entireBatchSize + miniBatchSize - 1) / miniBatchSize;
            var lastBlockSize = (nbBatchBlock==1)? entireBatchSize : (entireBatchSize % miniBatchSize);
            if (lastBlockSize == 0)
            {
                lastBlockSize = miniBatchSize;
            }
            _yPredictedBufferForMiniBatchGradientDescent = NewTensor(yExpected.Shape, _yPredictedBufferForMiniBatchGradientDescent, nameof(_yPredictedBufferForMiniBatchGradientDescent));
            int nbProcessed = 0;
            for (var blockId = 0; blockId < nbBatchBlock; blockId++)
            {
                var blockSize = (blockId == nbBatchBlock - 1) ? lastBlockSize : miniBatchSize;
                var xMiniBatch = X.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                var yExpectedMiniBatch = yExpected.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                var yPredictedMiniBatch = _yPredictedBufferForMiniBatchGradientDescent.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                
                Layers.Last().Set_y(yPredictedMiniBatch);
                var yPredictedMiniBatchV2 = Predict(xMiniBatch, isTraining);
                Debug.Assert(ReferenceEquals(yPredictedMiniBatch, yPredictedMiniBatchV2));
                if (isTraining)
                {
                    BackwardPropagation(yExpectedMiniBatch);
                    UpdateWeights(learningRateIfTraining);
                }
                if (!yPredictedMiniBatch.UseGPU)
                {
                    yPredictedMiniBatch.CopyTo(0, _yPredictedBufferForMiniBatchGradientDescent, _yPredictedBufferForMiniBatchGradientDescent.Idx(nbProcessed), yPredictedMiniBatch.Count);
                }
                nbProcessed += xMiniBatch.Shape[0];
            }
            return _yPredictedBufferForMiniBatchGradientDescent;
        }
        private Tensor NewTensorOfSameValue<T>(int[] shape, T sameValue, string description) where T: struct
        {
            var data = new T[Utils.Product(shape)];
            for (int i = 0; i < data.Length; ++i)
            {
                data[i] = sameValue;
            }
            if (Config.UseDoublePrecision)
            {
                return Config.UseGPU
                    ? new GPUTensor<T>(shape, data, description, Config.GpuWrapper)
                    : (Tensor) new CpuTensor<T>(shape, data, description);
            }
            else
            {
                return Config.UseGPU
                    ? (Tensor) new GPUTensor<T>(shape, data, description, Config.GpuWrapper)
                    : new CpuTensor<T>(shape, data, description);
            }
        }
    }
}
