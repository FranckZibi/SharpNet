﻿using System;
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
        private readonly ImageDataGenerator _imageDataGenerator;
        public NetworkConfig Config { get; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { get; set; } = "";
        private readonly Stopwatch _spInternalFit = new Stopwatch();
        private readonly Stopwatch _swUpdateWeights;
        private readonly Stopwatch _swPredictTraining;
        private readonly Stopwatch _swPredictNotTraining;
        private readonly Stopwatch _swBackwardPropagation;
        private readonly Stopwatch _swCreateInputForEpoch;
        private readonly Stopwatch _swComputeLossAndAccuracy;
        private readonly Stopwatch _swComputeLoss;
        private readonly Stopwatch _swComputeAccuracy;
        private Tensor _yPredictedBufferForMiniBatchGradientDescent;
        private readonly List<EpochData> _epochsData;
        #endregion

        public Network(NetworkConfig config, ImageDataGenerator imageDataGenerator = null, List<EpochData> epochData = null)
        {
            Config = config;
            _imageDataGenerator = imageDataGenerator??ImageDataGenerator.NoDataAugmentation;
            _epochsData = epochData ?? new List<EpochData>();
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
        }
        public void ClearMemory()
        {
            Config.GpuWrapper?.ClearMemory();
            Layers.ForEach(x => x?.Dispose());
            Layers.Clear();
            _epochsData.Clear();
        }

        #region network construction: adding layers
        public Network Input(int channelCount, int h, int w)
        {
            ClearMemory();
            Layers.Add(new InputLayer(channelCount, h, w, this));
            return this;
        }
        public Network Dense(int n_x, double _lambdaL2Regularization)
        {
            Debug.Assert(Layers.Count >= 1);
            var fullyConnectedLayer = new DenseLayer(n_x, _lambdaL2Regularization, this);
            Layers.Add(fullyConnectedLayer);
            return this;
        }
        public Network Convolution_BatchNorm(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return Convolution(filtersCount, f, stride, padding, lambdaL2Regularization)
                .BatchNorm();
        }
        public Network Convolution_BatchNorm_Activation(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Convolution_BatchNorm(filtersCount, f, stride, padding, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        public Network BatchNorm_Activation_Convolution(cudnnActivationMode_t activationFunction, int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return 
                BatchNorm()
                .Activation(activationFunction)
                .Convolution(filtersCount, f, stride, padding, lambdaL2Regularization);
        }
        public Network SumLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex)
        {
            Layers.Add(new SumLayer(previousIdentityLayerIndex, previousResidualLayerIndex, this));
            Debug.Assert(Layers[previousIdentityLayerIndex].SameOutputShape(Layers[previousResidualLayerIndex]));
            return this;
        }
        //add a shortcut from layer 'AddSumLayer' to current layer, adding a Conv Layer if necessary (for matching size)
        public Network Shortcut_IdentityConnection(int startOfBlockLayerIndex, int filtersCount, int stride, double lambdaL2Regularization)
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
                Convolution(filtersCount, 1, stride, 0, lambdaL2Regularization, startOfBlockLayerIndex);
                int convLayerIdInIdentityBlock = Layers.Last().LayerIndex;
                Layers.Add(new SumLayer(convLayerIdInIdentityBlock, previousResidualLayerIndex, this));
                Debug.Assert(Layers[convLayerIdInIdentityBlock].SameOutputShape(Layers[previousResidualLayerIndex]));
            }
            return this;
        }
        public Network Convolution_Activation_Pooling(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, cudnnActivationMode_t activationFunction, int poolingSize, int poolingStride)
        {
            return Convolution(filtersCount, f, stride, padding, lambdaL2Regularization)
                .Activation(activationFunction)
                .MaxPooling(poolingSize, poolingStride);
        }
        public Network Convolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization)
        {
            return Convolution(filtersCount, f, stride, padding, lambdaL2Regularization, Layers.Count - 1);
        }
        public Network Convolution(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, int previousLayerIndex)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(filtersCount, f, stride, padding, lambdaL2Regularization, previousLayerIndex, this));
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
        public Network Output(int n_x, double _lambdaL2Regularization, cudnnActivationMode_t activationFunctionType)
        {
            return Dense(n_x, _lambdaL2Regularization)
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

        public void Fit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, double learningRate, int numEpochs, int batchSize, CpuTensor<T> X_test = null, CpuTensor<T> Y_test = null) where T : struct
        {
            Fit(xCpu, yCpu, LearningRateScheduler.Constant(learningRate), null, numEpochs, batchSize, X_test, Y_test);
        }

        public void Fit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, ILearningRateScheduler lrScheduler, ReduceLROnPlateau reduceLROnPlateau, int numEpochs, int batchSize, CpuTensor<T> xTestCpu = null,CpuTensor<T> yTestCpu = null) where T : struct
        {
            var learningRateComputer = new LearningRateComputer(lrScheduler, reduceLROnPlateau, Config.MinimumLearningRate);
            Fit(xCpu, yCpu, learningRateComputer, numEpochs, batchSize, xTestCpu, yTestCpu);
        }
        private void Fit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, ILearningRateComputer learningRateComputer, int numEpochs, int batchSize, CpuTensor<T> xTestCpu = null, CpuTensor<T> yTestCpu = null) where T : struct
        {
            if (Config.UseDoublePrecision)
            {
                InternalFit(xCpu.ToDoublePrecision(), yCpu.ToDoublePrecision(), learningRateComputer, numEpochs, batchSize, xTestCpu?.ToDoublePrecision(), yTestCpu?.ToDoublePrecision());
            }
            else
            {
                InternalFit(xCpu.ToSinglePrecision(), yCpu.ToSinglePrecision(), learningRateComputer, numEpochs, batchSize, xTestCpu?.ToSinglePrecision(), yTestCpu?.ToSinglePrecision());
            }
        }
        //= ForwardPropagation
        public Tensor Predict(Tensor X, bool isTraining)
        {
            (isTraining ? _swPredictTraining : _swPredictNotTraining)?.Start();
            X = ReformatToCorrectType(X);
            ((InputLayer)Layers[0]).Set_y(X);
            foreach (var l in Layers.Skip(1))
            {
                l.ForwardPropagation(isTraining);
                //Info(Environment.NewLine+"Epoch:" + _indexCurrentEpochs + "; Layer:" + l.SummaryName() + "_" + l.LayerIndex + "; After ForwardPropagation:" + Environment.NewLine + l.ContentStats());
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
            _swBackwardPropagation?.Stop();
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
            var epochsData = (EpochData[])dicoFirstLine[nameof(_epochsData)];
            var network = new Network(config, imageDataGenerator, epochsData.ToList());
            for (int i = 1; i < content.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(content[i], network.Config.GpuWrapper), network));
            }
            return network;
        }


        private string Save(string path)
        {
            int indexLastCompletedEpoch = _epochsData.Count;
            var fileName = Path.Combine(path, "Network_" + Process.GetCurrentProcess().Id + "_" + indexLastCompletedEpoch + ".txt");
            var firstLine = new Serializer()
                .Add(Config.Serialize())
                .Add(_imageDataGenerator.Serialize())
                .Add(nameof(_epochsData), _epochsData.ToArray())
                .ToString();
            File.AppendAllLines(fileName, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(fileName, new[] { l.Serialize() });
            }
            return fileName;
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
        private void CheckInput<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, ILearningRateComputer learningRateComputer, int numEpochs, int batchSize, CpuTensor<T> X_testCpu, CpuTensor<T> Y_testCpu) where T : struct
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




        /*
        private double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch, ILearningRateScheduler lrScheduler, double learningRateMultiplicativeFactorFromReduceLrOnPlateau)
        {
            var learningRateFromScheduler = lrScheduler.LearningRate(epoch, blockIdInEpoch, nbBlocksInEpoch);
            var learningRateWithPlateauReduction = learningRateFromScheduler * learningRateMultiplicativeFactorFromReduceLrOnPlateau;
            learningRateWithPlateauReduction = Math.Max(learningRateWithPlateauReduction, Config.MinimumLearningRate);
            return learningRateWithPlateauReduction;
        }
        */

        public double FindBestLearningRate(Tensor x, Tensor y, int miniBatchSize)
        {
            _spInternalFit.Start();
        
            Info("Looking for best learning rate...");
            var learningRateFinder = new LearningRateFinder(miniBatchSize, x.Shape[0]);
            bool CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch, int blockId)
            {
                _swComputeLoss?.Start();
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction);
                _swComputeLoss?.Stop();
                return learningRateFinder.AddLossForLastBlockId(blockLoss);
            }

            MiniBatchGradientDescent(miniBatchSize, x, y, learningRateFinder, CallBackAfterEachMiniBatch);
            //TODO : restore weights at there original values
            //File.WriteAllText("c:/temp/ml/toto_"+DateTime.Now.Ticks+".csv", learningRateFinder.AsCsv());
            var bestLearningRate = learningRateFinder.BestLearningRate();
            Info("Best learning rate: "+ bestLearningRate);
            _spInternalFit.Stop();
            return bestLearningRate;
        }




        //here T is already of the target precision (double or float)
        private void InternalFit<T>(CpuTensor<T> xCpu, CpuTensor<T> yCpu, ILearningRateComputer learningRateComputer, int numEpochs, int miniBatchSize, CpuTensor<T> xTestCpu, CpuTensor<T> yTestCpu) where T : struct
        {
            Debug.Assert(Config.TypeSize == xCpu.TypeSize);
            Debug.Assert(Config.TypeSize == yCpu.TypeSize);
            Debug.Assert(learningRateComputer != null);
            _spInternalFit.Start();

            CheckInput(xCpu, yCpu, learningRateComputer, numEpochs, miniBatchSize, xTestCpu, yTestCpu);
            var yInputCpu = yCpu.Clone();
            var x = ReformatToCorrectDevice_GPU_or_CPU(xCpu);
            var y = ReformatToCorrectDevice_GPU_or_CPU(yCpu);
            var xTest = ReformatToCorrectDevice_GPU_or_CPU(xTestCpu);
            var yTest = ReformatToCorrectDevice_GPU_or_CPU(yTestCpu);
            Info(ToString());
            var blocksInEpoch = NbBlocksInEpoch(miniBatchSize, x.Shape[0]);
            if (Config.UseGPU)
            {
                LogDebug(Config.GpuWrapper.ToString());
            }
            LogDebug("Training Set: " + x + " => " + y);
            if (xTest != null)
            {
                LogDebug("Test Set: " + xTest + " => " + yTest);
            }
            Info("#Epochs=" + numEpochs + " BathSize=" + miniBatchSize);


            var enlargedXCpu = _imageDataGenerator.EnlargePictures(xCpu);
            var lastAutoSaveTime = DateTime.Now; //last time we saved the network
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


                #region Data augmentation
                if (epoch == 1)
                {
                    //for the very fist epoch we use exactly the same input, with no shuffling or data augmentation
                }
                else
                {
                    _swCreateInputForEpoch?.Start();
                    _imageDataGenerator.CreateInputForEpoch(enlargedXCpu, yInputCpu, xCpu, yCpu, Config.RandomizeOrder);
                    _swCreateInputForEpoch?.Stop();
                }
                if (x.UseGPU)
                {
                    ((GPUTensor<T>)x).CopyToDevice();
                    ((GPUTensor<T>)y).CopyToDevice();
                }
                #endregion

                #region Mini Batch gradient descent
                var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, blocksInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau);
                var yPredicted = MiniBatchGradientDescent(miniBatchSize, x, y, learningRateComputer);
                #endregion

                //We display stats about the just finished epoch
                _swComputeLossAndAccuracy?.Start();
                var trainLossAndAccuracy = ComputeLossAndAccuracy_From_Expected_vs_Predicted(y, yPredicted, Config.LossFunction);
                var lossAndAccuracyMsg = LossAndAccuracyToString(trainLossAndAccuracy, "");
                Tuple<double, double> validationLossAndAccuracy = null;
                if (xTest != null)
                {
                    //We compute the validation (= test) loss&accuracy
                    validationLossAndAccuracy = ComputeLossAndAccuracy(miniBatchSize, xTest, yTest);
                    lossAndAccuracyMsg += " - "+LossAndAccuracyToString(validationLossAndAccuracy, "val_");
                }
                _swComputeLossAndAccuracy?.Stop();
                double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                double nbStepsByEpoch = ((double)x.Shape[0]) / miniBatchSize;
                var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                Info("Epoch " + epoch + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRateAtEpochStart, 8)+" - "+lossAndAccuracyMsg);
                if (Config.UseGPU)
                {
                    LogDebug(Config.GpuWrapper.MemoryInfo());
                }
                if (Config.ProfileApplication)
                {
                    LogDebug(ProfilingComments());
                }

                #region we save the network in a file if necessary
                if (  ((epoch == numEpochs)&&(numEpochs>=10))
                    || (!string.IsNullOrEmpty(Config.AutoSavePath) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinuts) )
                {
                    var swSaveTime = Stopwatch.StartNew();
                    Info("Saving network in directory '"+Config.AutoSavePath+"' ...");
                    var fileName = Save(Config.AutoSavePath);
                    Info("Network saved in file '" + fileName + "' in "+ Math.Round(swSaveTime.Elapsed.TotalSeconds,1)+ "s");
                    lastAutoSaveTime = DateTime.Now;
                }
                #endregion

                #region we save stats about the just finished epoch
                var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, trainLossAndAccuracy.Item1, trainLossAndAccuracy.Item2, validationLossAndAccuracy?.Item1 ?? double.NaN, validationLossAndAccuracy?.Item2 ?? double.NaN, secondsForEpoch);
                _epochsData.Add(currentEpochData);
                #endregion
            }
            Info("Training for " + numEpochs + " epochs took: " + _spInternalFit.Elapsed.TotalSeconds + "s");
            if (!string.IsNullOrEmpty(Description))
            {
                LogDebug("Network Name: "+Description);
            }
            if (x.UseGPU)
            {
                x.Dispose();
            }
            _spInternalFit.Stop();
        }


        #region compute Loss and Accuracy
        //returns : Tuple<loss, accuracy>
        public Tuple<double, double> ComputeLossAndAccuracy(int miniBatchSize, Tensor x, Tensor y)
        {
            var yPredicted = MiniBatchGradientDescent(miniBatchSize, x, y);
            return ComputeLossAndAccuracy_From_Expected_vs_Predicted(y, yPredicted, Config.LossFunction);
        }
        private Tuple<double, double> ComputeLossAndAccuracy_From_Expected_vs_Predicted(Tensor yExpected, Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction)
        {
            _swComputeAccuracy?.Start();
            yExpected = ReformatToCorrectType(yExpected);
            yPredicted = ReformatToCorrectType(yPredicted);
            var countOk = yExpected.ComputeAccuracy(yPredicted);
            _swComputeAccuracy?.Stop();
            _swComputeLoss?.Start();
            var totalLoss = yExpected.ComputeLoss(yPredicted, lossFunction);
            _swComputeLoss?.Stop();
            return Tuple.Create(totalLoss, countOk / ((double)yExpected.Shape[0]));
        }
        private static string LossAndAccuracyToString(Tuple<double, double> lossAndAccuracy, string prefix)
        {
            return prefix + "loss: " + Math.Round(lossAndAccuracy.Item1, 4) + " - " + prefix + "acc: " + Math.Round(lossAndAccuracy.Item2, 4);
        }
        #endregion

        private Tensor ReformatToCorrectType(Tensor x)
        {
            x = ReformatToCorrectPrecision_float_or_double(x);
            x = ReformatToCorrectDevice_GPU_or_CPU(x);
            return x;
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
                if (Config.UseGPU)
                { 
                    result += ", CopyToDevice:" + Math.Round(100 * Config.GpuWrapper.SwCopyToDevice.ElapsedMilliseconds / totalMs, 0) + "%";
                    result += ", CopyToHost:" + Math.Round(100 * Config.GpuWrapper.SwCopyToHost.ElapsedMilliseconds / totalMs, 0) + "%";
                }
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
                //Info(Environment.NewLine + "Epoch:" + _indexCurrentEpochs + "; Layer:" + l.SummaryName() + "_" + l.LayerIndex + "; After UpdateWeights:" + Environment.NewLine+l.ContentStats());
            }
            _swUpdateWeights?.Stop();
        }
        private void Info(string msg) { Config.Logger.Info(msg); }
        private void LogDebug(string msg) { Config.Logger.Debug(msg); }

        /// <summary>
        /// Perform a mini batch gradient descent for an entire epoch, each mini batch will have 'miniBatchSize' elements
        /// </summary>
        /// <param name="miniBatchSize"></param>
        /// <param name="x">The input</param>
        /// <param name="y">Expected (target) output</param>
        /// <param name="learningRateComputerIfTraining">null if we are just using the network to predict the results (without updating weights)
        ///     not null if we need to update the weights between each mini batch</param>
        /// <param name="callBackToStop">Optional callback to be called at the end of each mini batch,
        ///     parameters are: 'mini batch expected output' + 'mini batch observed output' + 'current block Id'
        ///     If the callback returns true we should stop the computation</param>
        /// <returns>observed output associated with the input 'x'</returns>
        private Tensor MiniBatchGradientDescent(int miniBatchSize, Tensor x, Tensor y, ILearningRateComputer learningRateComputerIfTraining = null, Func<Tensor, Tensor, int, bool> callBackToStop = null)
        {
            x = ReformatToCorrectType(x);
            y = ReformatToCorrectType(y);
            bool isTraining = learningRateComputerIfTraining != null;
            var entireBatchSize = x.Shape[0];
            Debug.Assert(entireBatchSize == y.Shape[0]);
            if (miniBatchSize <= 0)
            {
                throw new Exception("invalid miniBatchSize size (" + miniBatchSize + ")");
            }
            int nbBatchBlock = NbBlocksInEpoch(miniBatchSize, entireBatchSize);
            var lastBlockSize = (nbBatchBlock==1)? entireBatchSize : (entireBatchSize % miniBatchSize);
            int epoch = _epochsData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(_epochsData) ?? 1.0;
            if (lastBlockSize == 0)
            {
                lastBlockSize = miniBatchSize;
            }
            _yPredictedBufferForMiniBatchGradientDescent = NewTensor(y.Shape, _yPredictedBufferForMiniBatchGradientDescent, nameof(_yPredictedBufferForMiniBatchGradientDescent));
            int nbProcessed = 0;

            for (var blockId = 0; blockId < nbBatchBlock; blockId++)
            {
                var blockSize = (blockId == nbBatchBlock - 1) ? lastBlockSize : miniBatchSize;
                var xMiniBatch = x.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                var yExpectedMiniBatch = y.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                var yPredictedMiniBatch = _yPredictedBufferForMiniBatchGradientDescent.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                Layers.Last().Set_y(yPredictedMiniBatch);
                Predict(xMiniBatch, isTraining);
                if (isTraining)
                {
                    BackwardPropagation(yExpectedMiniBatch);
                    UpdateWeights(learningRateComputerIfTraining.LearningRate(epoch, blockId, nbBatchBlock, lrMultiplicativeFactorFromReduceLrOnPlateau));
                }
                if (!yPredictedMiniBatch.UseGPU)
                {
                    yPredictedMiniBatch.CopyTo(0, _yPredictedBufferForMiniBatchGradientDescent, _yPredictedBufferForMiniBatchGradientDescent.Idx(nbProcessed), yPredictedMiniBatch.Count);
                }
                nbProcessed += xMiniBatch.Shape[0];
                if (callBackToStop != null && callBackToStop(yExpectedMiniBatch, yPredictedMiniBatch, blockId))
                {
                    break;
                }
            }
            return _yPredictedBufferForMiniBatchGradientDescent;
        }

        private static int NbBlocksInEpoch(int miniBatchSize, int entireBatchSize)
        {
            return (entireBatchSize + miniBatchSize - 1) / miniBatchSize;
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
