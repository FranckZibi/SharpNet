using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    //x                 (batchSize, x.C, x.H, x.W)
    public sealed class ConvolutionLayer : Layer
    {
        #region Private fields
        /// <summary>
        ///  Description of output shape ('y' tensor)
        /// if Depthwise Convolution:
        ///     (batchSize, _depthMultiplier*x.C, y.H, y.W)
        /// else
        ///     (batchSize, FiltersCount, y.H, y.W)
        ///     y.H = (x.H−F+2×pads) /Stride + 1
        ///     y.W = (x.W−F+2×pads) /Stride + 1
        /// </summary>
        private readonly bool _isDepthwiseConvolution;
        private readonly int _filtersCount;                     //only valid on default convolution   (_isDepthwiseConvolution=false)
        private readonly int _depthMultiplier;                  //only valid on depthwise convolution (_isDepthwiseConvolution=true)
        private readonly int _f;
        private readonly int _stride;
        private readonly PADDING_TYPE _paddingType;
        private readonly double _lambdaL2Regularization;
        private readonly Optimizer _optimizer;                  //Adam optimization or SGD optimization or null
        private bool UseBias => _convolutionBias != null;
        /// <summary>
        /// if Depthwise Convolution:
        ///     (_depthMultiplier, x.C, F, F)
        /// else
        ///     (FiltersCount, x.C, F, F)
        /// </summary>
        public Tensor Convolution { get; }

        /// <summary>
        /// if Depthwise Convolution:
        ///     (_depthMultiplier, x.C, 1, 1) or null is no bias should be used
        /// else
        ///     (1, FiltersCount, 1, 1) or null is no bias should be used
        /// </summary>
        private Tensor _convolutionBias;

        private Tensor _convolutionGradients;            // same shape as 'Convolution'
        /// <summary>
        /// same shape as '_convolutionBias'  or null is no bias should be used
        /// </summary>
        private Tensor _convolutionBiasGradients;
        #endregion


        public override Tensor Weights => Convolution;
        public override Tensor WeightGradients => _convolutionGradients;
        public override Tensor Bias => _convolutionBias;
        public override Tensor BiasGradients => _convolutionBiasGradients;

        public enum PADDING_TYPE { VALID, SAME}

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ConvolutionLayer(bool isDepthwiseConvolution, int filtersCount, int depthMultiplier, int f, int stride, PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, int previousLayerIndex, Network network, string layerName)
            : base(network, previousLayerIndex, layerName)
        {
            _isDepthwiseConvolution = isDepthwiseConvolution;
            _filtersCount = filtersCount;
            _depthMultiplier = depthMultiplier;
            if (_isDepthwiseConvolution && depthMultiplier != 1)
            {
                throw new NotImplementedException("only depthMultiplier=1 is supported in depthwise convolution");
            }
            _f = f;
            _stride = stride;
            _paddingType = paddingType;
            _lambdaL2Regularization = lambdaL2Regularization;

            //trainable params
            Convolution = GetNotInitializedFloatTensor(ConvolutionShape, nameof(Convolution));
            _convolutionBias = useBias
                ? GetNotInitializedFloatTensor(ConvolutionBiasShape, nameof(_convolutionBias))
                : null;

            _optimizer = Network.GetOptimizer(Convolution.Shape, _convolutionBias?.Shape);

            ResetWeights(false);
        }

        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (ConvolutionLayer)b;
            var equals = true;
            equals &= Utils.Equals(_isDepthwiseConvolution, other._isDepthwiseConvolution, id + ":_isDepthwiseConvolution", ref errors);
            equals &= Utils.Equals(_filtersCount, other._filtersCount, id + ":_filtersCount", ref errors);
            equals &= Utils.Equals(_depthMultiplier, other._depthMultiplier, id + ":_depthMultiplier", ref errors);
            equals &= Utils.Equals(_f, other._f, id+":_f", ref errors);
            equals &= Utils.Equals(_stride, other._stride, id + ":_stride", ref errors);
            equals &= Utils.Equals((int)_paddingType, (int)other._paddingType, id + ":_paddingType", ref errors);
            equals &= Utils.Equals(_lambdaL2Regularization, other._lambdaL2Regularization, epsilon, id + ":_lambdaL2Regularization", ref errors);
            equals &= _optimizer.Equals(other._optimizer, epsilon, id+":Optimizer", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return  RootSerializer() // 'RootSerializer()' will also serialize layer trainable params
                .Add(nameof(_isDepthwiseConvolution), _isDepthwiseConvolution).Add(nameof(_filtersCount), _filtersCount).Add(nameof(_depthMultiplier), _depthMultiplier)
                .Add(nameof(_f), _f).Add(nameof(_stride), _stride)
                .Add(nameof(_paddingType), (int)_paddingType)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(_optimizer.Serialize())
                .ToString();
        }
        public ConvolutionLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _isDepthwiseConvolution = (bool)serialized[nameof(_isDepthwiseConvolution)];
            _filtersCount = (int)serialized[nameof(_filtersCount)];
            _depthMultiplier = (int)serialized[nameof(_depthMultiplier)];
            _f = (int)serialized[nameof(_f)];
            _stride = (int)serialized[nameof(_stride)];
            _paddingType = (PADDING_TYPE)serialized[nameof(_paddingType)];
            _lambdaL2Regularization = (double)serialized[nameof(_lambdaL2Regularization)];

            //bias may be null if it has been disabled
            var useBias = serialized.ContainsKey(nameof(_convolutionBias));

            //trainable params
            Convolution = (Tensor)serialized[nameof(Convolution)];
            _convolutionBias = useBias ? (Tensor)serialized[nameof(_convolutionBias)] : null;

            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }
        #endregion
        public override int DisableBias()
        {
            int nbDisabledWeights = (_convolutionBias?.Count ?? 0);
            _convolutionBias?.Dispose();
            _convolutionBias = null;
            return nbDisabledWeights;
        }

        private Tensor _padded_X;

        /// <summary>
        /// compute y = x (conv) Convolution + ConvolutionBias
        /// </summary>
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(_padded_X == null);
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            Padding(x.Shape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);
            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                // cuDNN 7.x doesn't support asymmetric padding
                // we'll pad the input tensor 'x' so that we can use a symmetric padding
                StartForwardTimer(Type() + ">0Pad", isTraining);
                var paddedXShape = PaddedXShape(x.Shape, paddingTop, paddingBottom, paddingLeft, paddingRight);
                GetNotInitializedFloatTensor(ref _padded_X, paddedXShape, nameof(_padded_X));
                _padded_X.ZeroPadding(x, paddingTop, paddingBottom, paddingLeft, paddingRight);
                StopForwardTimer(Type() + ">0Pad", isTraining);
                StartForwardTimer(Type() + ">ConvAsym", isTraining);
                _padded_X.Convolution(Convolution, 0, 0, 0, 0, _stride, y, _isDepthwiseConvolution, ConvolutionAlgoPreference, Network.MemoryPool);
                if (!LayerOutputShouldBeKeptForBackwardPropagation(isTraining, FirstTrainableLayer(Network.Layers)))
                {
                    FreeMemory(ref _padded_X);
                }
                StopForwardTimer(Type() + ">ConvAsym", isTraining);
            }
            else
            {
                //symmetric padding
                StartForwardTimer(Type() + ">Conv", isTraining);
                x.Convolution(Convolution, paddingTop, paddingBottom, paddingLeft, paddingRight, _stride, y, _isDepthwiseConvolution, ConvolutionAlgoPreference, Network.MemoryPool);
                StopForwardTimer(Type() + ">Conv", isTraining);
            }

            if (UseBias)
            {
                StartForwardTimer(Type() + ">Bias", isTraining);
                _convolutionBias.BroadcastConvolutionBiasToOutput(y);
                StopForwardTimer(Type() + ">Bias", isTraining);
            }
        }

        
        private static int[] PaddedXShape(int[] xShape, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            return new[]{xShape[0], xShape[1], paddingTop + xShape[2] + paddingBottom, paddingLeft + xShape[3] + paddingRight};
        }

        public override int ExtraElementCountForForwardPropagation(int batchSize)
        {
            if (LayerIndex == 0)
            {
                return 0;
            }
            var xShape = PrevLayer.OutputShape(batchSize);
            Padding(xShape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);
            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                return Utils.Product(PaddedXShape(xShape, paddingTop, paddingBottom, paddingLeft, paddingRight));
            }
            return 0;
        }

        public override int ExtraElementCountForBackwardPropagation(int batchSize)
        {
            return ExtraElementCountForForwardPropagation(batchSize);
        }

        // dy => _convolutionGradients & _convolutionBiasGradients & dx
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            Debug.Assert(dx.Count == 1);

            //we allocate '_convolutionGradients' tensor
            GetNotInitializedFloatTensor(ref _convolutionGradients, Convolution.Shape, nameof(_convolutionGradients));

            if (UseBias)
            {
                //we allocate '_convolutionBiasGradients' tensor
                GetNotInitializedFloatTensor(ref _convolutionBiasGradients, _convolutionBias.Shape, nameof(_convolutionBiasGradients));
                //we compute '_convolutionBiasGradients'
                StartBackwardTimer(Type() + ">Bias");
                dy.ConvolutionBackwardBias(_convolutionBiasGradients);
                StopBackwardTimer(Type() + ">Bias");
            }

            // we compute '_convolutionGradients' (& dx if PrevLayer is not the input layer)
            Padding(x.Shape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);

            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                // cuDNN 7.x doesn't support asymmetric padding, we'll use the padded version of input tensor 'x'
                Debug.Assert(_padded_X != null);
                StartBackwardTimer(Type() + ">ConvAsym");
                var _padded_dX = GetNotInitializedFloatTensor(_padded_X.Shape, "_padded_dX");
                _padded_X.ConvolutionGradient(Convolution, dy, 0,0,0,0, _stride, _padded_dX, _convolutionGradients, _isDepthwiseConvolution, ConvolutionAlgoPreference, Network.MemoryPool);
                FreeMemory(ref _padded_X); //no more need of '_padded_X'
                StopBackwardTimer(Type() + ">ConvAsym");
                StartBackwardTimer(Type() + ">0Pad");
                dx[0]?.ZeroUnpadding(_padded_dX, paddingTop, paddingBottom, paddingLeft, paddingRight);
                FreeMemory(ref _padded_dX); //no more need of '_padded_dX'
                StopBackwardTimer(Type() + ">0Pad");
                Debug.Assert(_padded_X == null);
            }
            else
            {
                //symmetric padding
                Debug.Assert(_padded_X == null);
                StartBackwardTimer(Type() + ">Conv");
                x.ConvolutionGradient(Convolution, dy, paddingTop, paddingBottom, paddingLeft, paddingRight, _stride, dx[0], _convolutionGradients, _isDepthwiseConvolution, ConvolutionAlgoPreference, Network.MemoryPool);
                StopBackwardTimer(Type() + ">Conv");
            }

            if (UseL2Regularization)
            {
                var batchSize = dy.Shape[0];
                var alpha = 2 * batchSize * (float)_lambdaL2Regularization;
                _convolutionGradients.Update_Adding_Alpha_X(alpha, Convolution);
            }
        }

        public override void UpdateWeights(int batchSize, double learningRate)
        {
            if (Trainable)
            {
                _optimizer.UpdateWeights(learningRate, batchSize, Convolution, _convolutionGradients, _convolutionBias, _convolutionBiasGradients);
            }
            //no more need of '_convolutionGradients' and '_convolutionBiasGradients' : we can free them
            FreeMemory(ref _convolutionGradients);
            FreeMemory(ref _convolutionBiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            var fanIn = Convolution.MultDim0;
            var fanOut = Convolution.Shape[0];
            var stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));

            //trainable params
            Convolution.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, stdDev);
            _convolutionBias?.ZeroMemory();

            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override void Dispose()
        {
            base.Dispose();
            _optimizer?.Dispose();
        }

        public override string Type()
        {
            return _isDepthwiseConvolution? "DepthwiseConv2D" : "Conv2D";
        }

        public override string ToString()
        {
            var result = LayerName+": " + ShapeChangeDescription();
            result += " padding=" + _paddingType +" stride=" + _stride;
            result += " Filter"+ Utils.ShapeToString(Convolution?.Shape);
            result += (UseBias)?" with Bias":" no Bias";
            result += " ("+ MemoryDescription()+")";
            return result;
        }

        public override void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            var weightDatasetPath = DatasetNameToDatasetPath(_isDepthwiseConvolution ? "depthwise_kernel:0" : "kernel:0");
            h5FileDataset[weightDatasetPath].ChangeAxis(new[] { 3, 2, 0, 1 }).CopyTo(Convolution);
            //we load bias if necessary
            if (UseBias)
            {
                var biasDatasetPath = DatasetNameToDatasetPath(_isDepthwiseConvolution ? "depthwise_bias:0" : "bias:0");
                h5FileDataset[biasDatasetPath].CopyTo(_convolutionBias);
            }
        }

        public override int[] OutputShape(int batchSize)
        {
            var inputShape = PrevLayer.OutputShape(batchSize);
            var result = OutputShape(inputShape, Convolution.Shape, _paddingType, _stride, _isDepthwiseConvolution);
            Debug.Assert(result.Min() >= 1);
            return result;
        }



        /// <summary>
        /// Compute the output shape fo a convolution layer, given input shape 'inputShape' and convolution shape 'convolutionShape'
        /// </summary>
        /// <param name="inputShape">input shape: (batchSize, inputChannels, heightInput, widthInput)</param>
        /// <param name="convolutionShape">
        /// if isDepthwiseConvolution = true
        ///     convolution shape: (depthMultiplier, inputChannels, f, f)</param>
        /// else
        ///     convolution shape: (filtersCount, inputChannels, f, f)
        /// <param name="paddingType"></param>
        /// <param name="stride"></param>
        /// <param name="isDepthwiseConvolution"></param>
        /// <returns>output shape:
        /// if isDepthwiseConvolution = true
        ///     shape is (batchSize, inputChannels * depthMultiplier, heightOutput=H[heightInput], weightOutput=H[weightInput])
        ///else
        ///     shape is (batchSize, filtersCount, heightOutput=H[heightInput], weightOutput=H[weightInput])
        /// </returns>
        public static int[] OutputShape(int[] inputShape, int[] convolutionShape, PADDING_TYPE paddingType, int stride, bool isDepthwiseConvolution)
        {
            Debug.Assert(inputShape.Length == 4);
            Debug.Assert(convolutionShape.Length == 4);
            Debug.Assert(stride >= 1);
            Debug.Assert(inputShape[1] == convolutionShape[1]); //same channel depth for 'input shape' and 'convolution shape'
            Debug.Assert(convolutionShape[2] == convolutionShape[3]); //convolution height == convolution width
            var batchSize = inputShape[0];
            var inputChannels = inputShape[1];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            var f = convolutionShape[2];
            Debug.Assert(f % 2 == 1); // F must be odd
            var outputHeight = OutputLength(inputHeight, f, stride, paddingType);
            var outputWidth = OutputLength(inputWidth, f, stride, paddingType);
            int outputChannels = isDepthwiseConvolution
                ? inputChannels * convolutionShape[0]
                : convolutionShape[0];
            return new[] { batchSize, outputChannels, outputHeight, outputWidth };
        }

        /// <summary>
        /// //TODO add tests
        /// </summary>
        /// <param name="inputLength"></param>
        /// <param name="f"></param>
        /// <param name="stride"></param>
        /// <param name="paddingType"></param>
        /// <returns></returns>
        private static int OutputLength(int inputLength, int f, int stride, PADDING_TYPE paddingType)
        {
            switch (paddingType)
            {
                case PADDING_TYPE.VALID:  
                    return (inputLength - f) / stride + 1; 
                case PADDING_TYPE.SAME:
                    return (inputLength - 1) / stride + 1;
                default:
                    throw new NotImplementedException("unknown padding type "+paddingType);
            }
        }

        protected override List<Tensor> TensorsDependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { _padded_X};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        private void Padding(int[] xShape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight)
        {
            Debug.Assert(xShape.Length == 4);
            Padding(xShape[2], _f, _stride, _paddingType, Network.Config.CompatibilityMode, out paddingTop, out paddingBottom);
            Padding(xShape[3], _f, _stride, _paddingType, Network.Config.CompatibilityMode, out paddingLeft, out paddingRight);
        }

        public static bool IsAsymmetricPadding(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            return (paddingTop != paddingBottom || paddingLeft != paddingRight);
        }

        public static void Padding(int inputLength, int f, int stride, PADDING_TYPE paddingType, NetworkConfig.CompatibilityModeEnum compatibilityMode, out int paddingStart, out int paddingEnd)
        {
            int outputLength = OutputLength(inputLength, f, stride, paddingType);
            int totalPadding = Math.Max((outputLength - 1) * stride + f - inputLength, 0);
            switch (paddingType)
            {
                case PADDING_TYPE.VALID:
                    paddingStart = paddingEnd = 0;
                    return;
                case PADDING_TYPE.SAME:

                    if (compatibilityMode == NetworkConfig.CompatibilityModeEnum.TensorFlow1 || compatibilityMode == NetworkConfig.CompatibilityModeEnum.TensorFlow2)
                    {
                        //see: https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
                        paddingStart = totalPadding / 2;
                        paddingEnd = totalPadding - paddingStart;
                    }
                    else
                    {
                        //TODO check formula
                        paddingStart = (totalPadding + 1) / 2;
                        paddingEnd = paddingStart;
                    }
                    return;
                default:
                    throw new NotImplementedException("unknown padding type " + paddingType);
            }
        }



        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;

        protected override List<Tensor> TrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { Convolution, _convolutionBias};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        protected override List<Tensor> NonTrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor>();
                if (_optimizer != null)
                {
                    result.AddRange(_optimizer.EmbeddedTensors);
                }
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        private int[] ConvolutionShape
        {
            get
            {
                var channels = PrevLayer.OutputShape(1)[1];
                return _isDepthwiseConvolution
                    ?new[] { _depthMultiplier, channels, _f, _f }
                    :new[] { _filtersCount, channels, _f, _f };
            }
        }

        private int[] ConvolutionBiasShape
        {
            get
            {
                if (_isDepthwiseConvolution)
                {
                    var inputChannels = PrevLayer.OutputShape(1)[1];
                    return new[] { _depthMultiplier, inputChannels, 1, 1 };
                }
                else
                {
                    return new[] { 1, _filtersCount, 1, 1 };
                }
            }
        }
        private GPUWrapper.ConvolutionAlgoPreference ConvolutionAlgoPreference => Network.Config.ConvolutionAlgoPreference;
    }
}
