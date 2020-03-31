using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    //x                 (batchSize, x.C, x.H, x.W)
    public sealed class ConvolutionLayer : Layer
    {
        #region Private fields
        /// <summary>
        /// if Depthwise Convolution:
        ///     (batchSize, _depthMultiplier*x.C, y.H, y.W)
        /// else
        ///     (batchSize, FiltersCount, y.H, y.W)
        ///     y.H = (x.H−F+2×pads) /Stride + 1
        ///     y.W = (x.W−F+2×pads) /Stride + 1
        /// </summary>
        public override Tensor y { get; protected set; }
        private readonly bool _isDepthwiseConvolution;
        private readonly int _filtersCount;                     //only valid on default convolution   (_isDepthwiseConvolution=false)
        private readonly int _depthMultiplier;                  //only valid on depthwise convolution (_isDepthwiseConvolution=true)
        private readonly int _f;
        private readonly int _stride;
        private readonly PADDING_TYPE _paddingType;
        private readonly double _lambdaL2Regularization;
        private readonly Optimizer _optimizer;                  //Adam optimization or SGD optimization or null
        private bool UseBias => ConvolutionBias != null;
        /// <summary>
        /// if Depthwise Convolution:
        ///     (_depthMultiplier, x.C, F, F)
        /// else
        ///     (FiltersCount, x.C, F, F)
        /// </summary>
        public Tensor Convolution { get; }
        public Tensor ConvolutionGradients { get; }            // same as 'Convolution'
        /// <summary>
        /// if Depthwise Convolution:
        ///     (_depthMultiplier, x.C, 1, 1) or null is no bias should be used
        /// else
        ///     (1, FiltersCount, 1, 1) or null is no bias should be used
        /// </summary>
        public Tensor ConvolutionBias { get; private set; }
        public Tensor ConvolutionBiasGradients { get; private set; }        // same as 'ConvolutionBias'  or null is no bias should be used
        #endregion

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
            Convolution = Network.NewNotInitializedTensor(ConvolutionShape, nameof(Convolution));
            ConvolutionGradients = Network.NewNotInitializedTensor(Convolution.Shape, nameof(ConvolutionGradients));
            if (useBias)
            {
                ConvolutionBias = Network.NewNotInitializedTensor(ConvolutionBiasShape, nameof(ConvolutionBias));
                ConvolutionBiasGradients = Network.NewNotInitializedTensor(ConvolutionBias.Shape, nameof(ConvolutionBiasGradients));
            }
            _optimizer = Network.GetOptimizer(Convolution.Shape, ConvolutionBias?.Shape);
            ResetWeights(false);
        }

        public override Layer Clone(Network newNetwork) { return new ConvolutionLayer(this, newNetwork); }
        private ConvolutionLayer(ConvolutionLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _isDepthwiseConvolution = toClone._isDepthwiseConvolution;
            _filtersCount = toClone._filtersCount;
            _depthMultiplier = toClone._depthMultiplier;
            _f = toClone._f;
            _stride = toClone._stride;
            _paddingType = toClone._paddingType;
            _lambdaL2Regularization = toClone._lambdaL2Regularization;
            Convolution = toClone.Convolution?.Clone(newNetwork.GpuWrapper);
            ConvolutionGradients = toClone.ConvolutionGradients?.Clone(newNetwork.GpuWrapper);
            ConvolutionBias = toClone.ConvolutionBias?.Clone(newNetwork.GpuWrapper);
            ConvolutionBiasGradients = toClone.ConvolutionBiasGradients?.Clone(newNetwork.GpuWrapper);
            _optimizer = toClone._optimizer?.Clone(newNetwork);
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
            return  RootSerializer()
                .Add(nameof(_isDepthwiseConvolution), _isDepthwiseConvolution).Add(nameof(_filtersCount), _filtersCount).Add(nameof(_depthMultiplier), _depthMultiplier)
                .Add(nameof(_f), _f).Add(nameof(_stride), _stride)
                .Add(nameof(_paddingType), (int)_paddingType)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(Convolution).Add(ConvolutionGradients)
                .Add(ConvolutionBias).Add(ConvolutionBiasGradients)
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
            Convolution = (Tensor)serialized[nameof(Convolution)];
            ConvolutionGradients = (Tensor)serialized[nameof(ConvolutionGradients)];
            //Bias may be null if it has been disabled
            ConvolutionBias = serialized.TryGet<Tensor>(nameof(ConvolutionBias));
            ConvolutionBiasGradients = serialized.TryGet<Tensor>(nameof(ConvolutionBiasGradients));
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }
        #endregion
        public override int DisableBias()
        {
            int nbDisabledWeights = (ConvolutionBias?.Count ?? 0) + (ConvolutionBiasGradients?.Count ?? 0);
            ConvolutionBias?.Dispose();
            ConvolutionBias = null;
            ConvolutionBiasGradients?.Dispose();
            ConvolutionBiasGradients = null;
            return nbDisabledWeights;
        }


        private Tensor _paddedX;


        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x (conv) Convolution + ConvolutionBias

            Padding(out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);

            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                // cuDNN 7.x doesn't support asymmetric padding
                // we'll pad the input tensor 'x' so that we can use a symmetric padding
                var paddedXShape = new[]{x.Shape[0], x.Shape[1], paddingTop+x.Shape[2]+paddingBottom, paddingLeft+x.Shape[3]+paddingRight};
                _paddedX = Network.NewNotInitializedTensor(paddedXShape, _paddedX, nameof(_paddedX));
                _paddedX.ZeroPadding(x, paddingTop, paddingBottom, paddingLeft, paddingRight);
                _paddedX.Convolution(Convolution, 0, 0, 0, 0, _stride, y, _isDepthwiseConvolution);
            }
            else
            {
                //symmetric padding
                x.Convolution(Convolution, paddingTop, paddingBottom, paddingLeft, paddingRight, _stride, y, _isDepthwiseConvolution);
            }

            if (UseBias)
            {
                ConvolutionBias.BroadcastConvolutionBiasToOutput(y);
            }
        }


        // dy => ConvolutionGradient & dx
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            Debug.Assert(ConvolutionBias == null || ConvolutionBias.SameShape(ConvolutionBiasGradients));

            // we compute ConvolutionBiasGradients
            if (UseBias)
            {
                dy.ConvolutionBackwardBias(ConvolutionBiasGradients);
            }

            // we compute ConvolutionGradient (& dx if PrevLayer is not the input layer)
            var x = PrevLayer.y;
            Padding(out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);

            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                // cuDNN 7.x doesn't support asymmetric padding, we'll use the padded version of input tensor 'x'
                _paddedX.ConvolutionGradient(Convolution, dy, 0,0,0,0, _stride, dx[0], ConvolutionGradients, _isDepthwiseConvolution);
            }
            else
            {
                //symmetric padding
                x.ConvolutionGradient(Convolution, dy, paddingTop, paddingBottom, paddingLeft, paddingRight, _stride, dx[0], ConvolutionGradients, _isDepthwiseConvolution);
            }

            if (UseL2Regularization)
            {
                var batchSize = y.Shape[0];
                var alpha = 2 * batchSize * (float)_lambdaL2Regularization;
                ConvolutionGradients.Update_Adding_Alpha_X(alpha, Convolution);
            }
        }
        public override void UpdateWeights(double learningRate)
        {
            if (!Trainable)
            {
                return;
            }
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Convolution, ConvolutionGradients, ConvolutionBias, ConvolutionBiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            var fanIn = Convolution.MultDim0;
            var fanOut = Convolution.Shape[0];
            var stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));
            Convolution.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, stdDev);
            ConvolutionGradients .ZeroMemory();
            ConvolutionBias?.ZeroMemory();
            ConvolutionBiasGradients?.ZeroMemory();
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override int TotalParams => Convolution.Count + (ConvolutionBias?.Count??0);

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


        public override void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset)
        {
            string weightDatasetName = _isDepthwiseConvolution ? "depthwise_kernel:0" : "kernel:0";
            LoadFromH5Dataset(h5FileDataset, weightDatasetName, Convolution, new[] { 3, 2, 1, 0 });
            //we load bias if necessary
            string biasDatasetName = _isDepthwiseConvolution ? "depthwise_bias:0" : "bias:0";
            var biasDatasetPath = DatasetNameToDatasetPath(biasDatasetName);
            if (UseBias)
            {
                var cpuTensor = (CpuTensor<float>)h5FileDataset[biasDatasetPath];
                var reshapedCpuTensor = cpuTensor.WithNewShape(new[] { 1, cpuTensor.Shape[0], 1, 1 });
                LoadFromH5Dataset(reshapedCpuTensor, ConvolutionBias);
            }
            else
            {
                //we ensure that no bias was provided in input
                if (h5FileDataset.ContainsKey(biasDatasetPath))
                {
                    throw new ArgumentException("found bias tensor " + biasDatasetPath + "for layer " + LayerName + " although bias is disabled");

                }
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
        public static int OutputLength(int inputLength, int f, int stride, PADDING_TYPE paddingType)
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
                var result = new List<Tensor> { y, _paddedX };
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        private void Padding(out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight)
        {
            var x = PrevLayer.y;
            Padding(x.Shape[2], _f, _stride, _paddingType, Network.Config.CompatibilityMode, out paddingTop, out paddingBottom);
            Padding(x.Shape[3], _f, _stride, _paddingType, Network.Config.CompatibilityMode, out paddingLeft, out paddingRight);
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
        public override List<Tensor> TensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { Convolution, ConvolutionGradients, ConvolutionBias, ConvolutionBiasGradients };
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
    }
}
