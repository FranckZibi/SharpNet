using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Layers
{
    /// <summary>
    /// Input 'x' shape:
    ///     (batchSize, x.C, x.H, x.W)                      if _isConv1D == false
    ///     (batchSize, inputSize, timeSteps)               if _isConv1D == true
    /// Output 'y' tensor shape:
    ///     (batchSize, depth_multiplier*x.C, y.H, y.W)      if Depthwise Convolution 2D
    ///     (batchSize, out_channels, y.H, y.W)             if Standard Convolution 2D
    ///             y.H = (x.H−kernel_height+2×pads) /Stride + 1
    ///             y.W = (x.W−kernel_width+2×pads) /Stride + 1
    ///     (batchSize, out_channels, newTimeSteps)         if Standard Convolution 1D
    ///             newTimeSteps = (timeSteps−kernel_width+2×pads) /Stride + 1
    /// </summary>
    public sealed class ConvolutionLayer : Layer
    {
        #region Private fields
        #region trainable parameters

        /// <summary>
        /// if Depthwise Convolution:
        ///     (depth_multiplier, x.C, kernel_height, kernel_width)
        /// else
        ///     (out_channels, x.C, kernel_height, kernel_width)
        /// </summary>
        [NotNull] private Tensor _convolution;

        /// <summary>
        /// if Depthwise Convolution:
        ///     (depth_multiplier, x.C, 1, 1) or null is no bias should be used
        /// else
        ///     (1, out_channels, 1, 1) or null is no bias should be used
        /// </summary>
        [CanBeNull] private Tensor _convolutionBias;
        #endregion
        #region gradients
        [NotNull] private Tensor _convolutionGradients;            // same shape as 'Convolution'
        /// <summary>
        /// same shape as '_convolutionBias'  or null is no bias should be used
        /// </summary>
        [CanBeNull] private Tensor _convolutionBiasGradients;
        #endregion
        private readonly bool _isDepthwiseConvolution;
        private readonly bool _isConv1D;
        private readonly int out_channels;                     //only valid on default convolution   (_isDepthwiseConvolution=false)
        private readonly int depth_multiplier;                  //only valid on depthwise convolution (_isDepthwiseConvolution=true)
        private readonly int kernel_height;
        private readonly int kernel_width;
        private readonly int stride;
        private readonly PADDING_TYPE _paddingType;
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGF
        /// </summary>
        [NotNull] private readonly Optimizer _optimizer;
        private Tensor _padded_X;
        #endregion

        public enum PADDING_TYPE { VALID, SAME, CAUSAL}

        /// <summary>
        /// No need to configure the number of channels by filter: it is always the same as in previous layer
        /// </summary>
        public ConvolutionLayer(bool isDepthwiseConvolution, bool isConv1D, int out_channels, int depth_multiplier, int kernel_height, int kernel_width,int stride, PADDING_TYPE paddingType, bool bias, int previousLayerIndex, bool trainable, Network network, string layerName)
            : base(network, new[] { previousLayerIndex}, layerName)
        {
            if (isDepthwiseConvolution)
            {
                if (depth_multiplier != 1)
                {
                    throw new NotImplementedException("only depth_multiplier=1 is supported in depthwise convolution");
                }
            }
            if (isConv1D && kernel_height != 1)
            {
                throw new NotImplementedException($"{nameof(kernel_height)} must be 1 if Conv1D (found: {kernel_height})");
            }
            _isDepthwiseConvolution = isDepthwiseConvolution;
            _isConv1D = isConv1D;

            this.out_channels = out_channels;
            this.depth_multiplier = depth_multiplier;
            this.kernel_height = kernel_height;
            this.kernel_width = kernel_width;
            this.stride = stride;
            _paddingType = paddingType;
            Trainable = trainable;

            //trainable params
            _convolution = GetFloatTensor(ConvolutionShape);
            _convolutionBias = bias ? GetFloatTensor(ConvolutionBiasShape) : null;

            //gradients
            _convolutionGradients = GetFloatTensor(_convolution.Shape);
            _convolutionBiasGradients = (_convolutionBias!=null)? GetFloatTensor(_convolutionBias.Shape) : null;

            _optimizer = Sample.GetOptimizer(_convolution.Shape, _convolutionBias?.Shape, MemoryPool);

            ResetParameters(false);
        }

        #region forward and backward propagation
        /// <summary>
        /// compute y = x (conv) Convolution + ConvolutionBias
        /// </summary>
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(_padded_X == null);
            Debug.Assert(allX.Count == 1);
            var x = allX[0];

            var x4D = _isConv1D ? Conv1d_to_Conv2D(x) : x;
            Debug.Assert(x4D.Shape.Length == 4);
            var y4D = _isConv1D ? Conv1d_to_Conv2D(y) : y;
            Debug.Assert(y4D.Shape.Length == 4);

            Padding(x4D.Shape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);
            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                //Debug.Assert(!_isConv1D);

                // cuDNN 7.x doesn't support asymmetric padding
                // we'll pad the input tensor 'x' so that we can use a symmetric padding
                StartForwardTimer(LayerType() + ">ConvAsym", isTraining);
                var paddedXShape = PaddedXShape(x4D.Shape, paddingTop, paddingBottom, paddingLeft, paddingRight);
                GetFloatTensor(ref _padded_X, paddedXShape);
                _padded_X.ZeroPadding(x4D, paddingTop, paddingBottom, paddingLeft, paddingRight);
                _padded_X.Convolution(_convolution, 0, 0, 0, 0, stride, y4D, _isDepthwiseConvolution, ConvolutionAlgoPreference, MemoryPool);
                if (PreviousLayers.All(l=>!l.LayerOutputShouldBeKeptForBackwardPropagation(isTraining)))
                {
                    FreeFloatTensor(ref _padded_X);
                }
                StopForwardTimer(LayerType() + ">ConvAsym", isTraining);
            }
            else
            {
                //symmetric padding
                StartForwardTimer(LayerType() + ">Conv", isTraining);
                x4D.Convolution(_convolution, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, y4D, _isDepthwiseConvolution, ConvolutionAlgoPreference, MemoryPool);
                StopForwardTimer(LayerType() + ">Conv", isTraining);
            }

            StartForwardTimer(LayerType() + ">Bias", isTraining);
            _convolutionBias?.BroadcastConvolutionBiasToOutput(y4D);
            StopForwardTimer(LayerType() + ">Bias", isTraining);
        }
        /// <summary>
        /// dy => _convolutionGradients & _convolutionBiasGradients & dx 
        /// </summary>
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> alldX)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(alldX.Count == 1);
            var x = allX[0];
            var dx = alldX[0];

            var x4D = _isConv1D ? Conv1d_to_Conv2D(x) : x;
            var dx4D = _isConv1D ? Conv1d_to_Conv2D(dx) : dx;
            var dy4D = _isConv1D ? Conv1d_to_Conv2D(dy) : dy;

            Debug.Assert(x4D.Shape.Length == 4);
            Debug.Assert(dy4D.Shape.Length == 4);

            if (bias)
            {
                Debug.Assert(_convolutionBiasGradients != null);
                //we compute '_convolutionBiasGradients'
                StartBackwardTimer(LayerType() + ">Bias");
                dy4D.ConvolutionBackwardBias(_convolutionBiasGradients);
                StopBackwardTimer(LayerType() + ">Bias");
            }

            // we compute '_convolutionGradients' (& dx if PrevLayer is not the input layer)
            Padding(x4D.Shape, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight);

            if (IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
            {
                // cuDNN 7.x doesn't support asymmetric padding, we'll use the padded version of input tensor 'x'
                StartBackwardTimer(LayerType() + ">ConvAsym");
                Debug.Assert(_padded_X != null);
                var _padded_dX = GetFloatTensor(_padded_X.Shape);
                _padded_X.ConvolutionGradient(_convolution, dy4D, 0, 0, 0, 0, stride, _padded_dX, _convolutionGradients, _isDepthwiseConvolution, ConvolutionAlgoPreference, MemoryPool);
                FreeFloatTensor(ref _padded_X); //no more need of '_padded_X'
                dx4D?.ZeroUnpadding(_padded_dX, paddingTop, paddingBottom, paddingLeft, paddingRight);
                FreeFloatTensor(ref _padded_dX); //no more need of '_padded_dX'
                Debug.Assert(_padded_X == null);
                StopBackwardTimer(LayerType() + ">ConvAsym");
            }
            else
            {
                //symmetric padding
                StartBackwardTimer(LayerType() + ">Conv");
                Debug.Assert(_padded_X == null);
                x4D.ConvolutionGradient(_convolution, dy4D, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, dx4D, _convolutionGradients, _isDepthwiseConvolution, ConvolutionAlgoPreference, MemoryPool);
                StopBackwardTimer(LayerType() + ">Conv");
            }

            if (Sample.Use_weight_decay_in_backpropagation)
            {
                var batchSize = dy4D.Shape[0];
                var weight_decay_mulitplier = 2;
                if (Sample.CompatibilityMode == CompatibilityModeEnum.PyTorch) { weight_decay_mulitplier = 1; }
                var alpha = weight_decay_mulitplier * batchSize * (float)Sample.weight_decay;
                _convolutionGradients.Update_Adding_Alpha_X(alpha, _convolution);
                if (Sample.CompatibilityMode == CompatibilityModeEnum.PyTorch && bias)
                {
                    Debug.Assert(_convolutionBiasGradients != null);
                    _convolutionBiasGradients.Update_Adding_Alpha_X(alpha, _convolutionBias);
                }
            }
        }

        private static int[] ReshapeConv1d_to_Conv2D(int[] shape)
        {
            Debug.Assert(shape.Length == 3);
            return new[] { shape[0], shape[1], 1, shape[2] };
        }
        private static Tensor Conv1d_to_Conv2D(Tensor t)
        {
            return t?.Reshape(ReshapeConv1d_to_Conv2D(t.Shape));
        }
        private static int[] ReshapeConv2d_to_Conv1D(int[] shape)
        {
            Debug.Assert(shape.Length == 4);
            Debug.Assert(shape[2] == 1);
            return new[] { shape[0], shape[1], shape[3] };
        }


        public override bool OutputNeededForBackwardPropagation => false;
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _convolution;
        public override Tensor WeightGradients => _convolutionGradients;
        public override Tensor Bias => _convolutionBias;
        public override Tensor BiasGradients => _convolutionBiasGradients;
        protected override Optimizer Optimizer => _optimizer;

        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                             {
                                 Tuple.Create(_convolution, ConvolutionDatasetPath), 
                                 Tuple.Create(_convolutionBias, ConvolutionBiasDatasetPath)
                             };
                result.RemoveAll(t => t.Item1 == null);
                return result;
            }
        }
        public override int DisableBias()
        {
            int nbDisabledWeights = (_convolutionBias?.Count ?? 0);
            FreeFloatTensor(ref _convolutionBias);
            FreeFloatTensor(ref _convolutionBiasGradients);
            return nbDisabledWeights;
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            var fanIn = _convolution.MultDim0;
            var fanOut = _convolution.Shape[0];
            var stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));

            //trainable params
            _convolution.NormalDistribution(Rand, 0.0 /* mean */, stdDev);
            _convolutionBias?.ZeroMemory();

            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override void ReplaceParameters(List<Tensor> newParameters)
        {
            FreeFloatTensor(ref _convolution);
            _convolution = newParameters[0];
            if (_convolutionBias != null)
            {
                Debug.Assert(newParameters.Count == 2);
                FreeFloatTensor(ref _convolutionBias);
                _convolutionBias = newParameters[1];
            }
            else
            {
                Debug.Assert(newParameters.Count == 1);
            }
        }
        public override void LoadParameters(IDictionary<string, Tensor> h5FileDataset, CompatibilityModeEnum originFramework)
        {
            if (originFramework == CompatibilityModeEnum.TensorFlow)
            {
                h5FileDataset[ConvolutionDatasetPath].ChangeAxis(new[] { 3, 2, 0, 1 }).CopyTo(_convolution);
            }
            else
            {
                h5FileDataset[ConvolutionDatasetPath].CopyTo(_convolution);
            }
            if (bias)
            {
                h5FileDataset[ConvolutionBiasDatasetPath].CopyTo(_convolutionBias);
            }
        }
        public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(CompatibilityModeEnum originFramework)
        {
            var result = new Dictionary<string, CpuTensor<float>>();
            var weights = _convolution.ToCpuFloat();
            if (originFramework == CompatibilityModeEnum.TensorFlow)
            {
                weights = (CpuTensor<float>)weights.ChangeAxis(new[] { 2, 3, 1, 0 });
            }
            else
            {
                if (_isConv1D)
                {
                    weights = (CpuTensor<float>)weights.Reshape(weights.Shape[0], weights.Shape[1], -1);
                }
                if (_isDepthwiseConvolution)
                {
                    weights = (CpuTensor<float>)weights.Reshape(-1, 1, weights.Shape[2], weights.Shape[3]);
                }
            }
            result.Add(ConvolutionDatasetPath, weights);
            if (bias)
            {
                var bias = _convolutionBias;
                if (originFramework != CompatibilityModeEnum.TensorFlow)
                {
                    bias = bias.Reshape(-1);
                }
                // ReSharper disable once PossibleNullReferenceException
                result.Add(ConvolutionBiasDatasetPath, bias.ToCpuFloat());
            }
            return result;
        }

        public override void ReplaceGradients(List<Tensor> newGradients)
        {
            FreeFloatTensor(ref _convolutionGradients);
            _convolutionGradients = newGradients[0];
            if (_convolutionBiasGradients != null)
            {
                Debug.Assert(newGradients.Count == 2);
                FreeFloatTensor(ref _convolutionBiasGradients);
                _convolutionBiasGradients = newGradients[1];
            }
            else
            {
                Debug.Assert(newGradients.Count == 1);
            }
        }
        private string ConvolutionDatasetPath => DatasetNameToDatasetPath("weight");
        private string ConvolutionBiasDatasetPath => DatasetNameToDatasetPath("bias");
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_isDepthwiseConvolution), _isDepthwiseConvolution)
                .Add(nameof(_isConv1D), _isConv1D)
                .Add(nameof(out_channels), out_channels)
                .Add(nameof(depth_multiplier), depth_multiplier)
                .Add(nameof(kernel_height), kernel_height)
                .Add(nameof(kernel_width), kernel_width)
                .Add(nameof(stride), stride)
                .Add(nameof(_paddingType), _paddingType.ToString())
                .Add(nameof(bias), bias)
                .ToString();
        }
        public static ConvolutionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            int kernel_height = serialized.ContainsKey(nameof(kernel_height)) ? (int)serialized[nameof(kernel_height)] : (int)serialized["_f"];
            int kernel_width = serialized.ContainsKey(nameof(kernel_width)) ? (int)serialized[nameof(kernel_width)] : (int)serialized["_f"];
            bool isConv1D = serialized.ContainsKey(nameof(_isConv1D)) ? (bool)serialized[nameof(_isConv1D)] : false;

            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new ConvolutionLayer(
                (bool) serialized[nameof(_isDepthwiseConvolution)],
                isConv1D,
                (int) serialized[nameof(out_channels)],
                (int) serialized[nameof(depth_multiplier)],
                kernel_height,
                kernel_width,
                (int)serialized[nameof(stride)],
                (PADDING_TYPE)Enum.Parse(typeof(PADDING_TYPE), (string)serialized[nameof(_paddingType)]),
                (bool)serialized[nameof(bias)],
                previousLayerIndexes[0],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
        private int PreviousLayerIndex => PreviousLayerIndexes[0];
        public override string LayerType()
        {
            if (_isConv1D)
            {
                return "Conv1D";
            }
            return _isDepthwiseConvolution? "DepthwiseConv2D" : "Conv2D";
        }
        public override string ToString()
        {
            var result = LayerName+": " + ShapeChangeDescription();
            result += " padding=" + _paddingType +" stride=" + stride;
            result += " Filter"+ Utils.ShapeToString(_convolution.Shape);
            result += (bias)?" with Bias":" no Bias";
            result += " ("+ TotalParams+" neurons)";
            return result;
        }
      
        public override int[] OutputShape(int batchSize)
        {
            var inputShape = PrevLayer.OutputShape(batchSize);
            var result = OutputShape(inputShape, _convolution.Shape, _paddingType, stride, _isDepthwiseConvolution);
            Debug.Assert(result.Min() >= 1);
            return result;
        }
        /// <summary>
        /// Compute the output shape fo a convolution layer, given input shape 'inputShape' and convolution shape 'convolutionShape'
        /// </summary>
        /// <param name="inputShape">input shape: (batchSize, input_channels, heightInput, widthInput)</param>
        /// <param name="convolutionShape">
        /// if isDepthwiseConvolution = true
        ///     convolution shape: (depth_multiplier, x.C, kernel_height, kernel_width)</param>
        /// else
        ///     convolution shape: (out_channels, x.C, kernel_height, kernel_width)
        /// <param name="paddingType"></param>
        /// <param name="stride"></param>
        /// <param name="isDepthwiseConvolution"></param>
        /// <returns>output shape:
        /// if isDepthwiseConvolution = true
        ///     shape is (batchSize, x.C * depth_multiplier, heightOutput=H[heightInput], weightOutput=H[weightInput])
        ///else
        ///     shape is (batchSize, out_channels, heightOutput=H[heightInput], weightOutput=H[weightInput])
        /// </returns>
        public static int[] OutputShape(int[] inputShape, int[] convolutionShape, PADDING_TYPE paddingType, int stride, bool isDepthwiseConvolution)
        {
            if (inputShape.Length == 3)
            {
                //Conv1D
                Debug.Assert(!isDepthwiseConvolution);
                var inputShapeConv2D = ReshapeConv1d_to_Conv2D(inputShape);
                // ReSharper disable once ConditionIsAlwaysTrueOrFalse
                var outputShapeConv2D = OutputShape(inputShapeConv2D, convolutionShape, paddingType, stride, isDepthwiseConvolution);
                return ReshapeConv2d_to_Conv1D(outputShapeConv2D);
            }

            Debug.Assert(inputShape.Length == 4);
            Debug.Assert(convolutionShape.Length == 4);
            Debug.Assert(stride >= 1);
            Debug.Assert(inputShape[1] == convolutionShape[1]); //same channel depth for 'input shape' and 'convolution shape'
            //Debug.Assert(convolutionShape[2] == convolutionShape[3]); //convolution height == convolution width
            var batchSize = inputShape[0];
            var input_channels = inputShape[1];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            var kernel_height = convolutionShape[2];
            Debug.Assert(kernel_height % 2 == 1); // kernel_height must be odd
            var kernel_width = convolutionShape[3];
            Debug.Assert(kernel_width % 2 == 1); // kernel_width must be odd
            var outputHeight = OutputLength(inputHeight, kernel_height, stride, paddingType);
            var outputWidth = OutputLength(inputWidth, kernel_width, stride, paddingType);
            int outputChannels = isDepthwiseConvolution
                ? input_channels * convolutionShape[0]
                : convolutionShape[0];
            return new[] { batchSize, outputChannels, outputHeight, outputWidth };
        }
        public static bool IsAsymmetricPadding(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            return (paddingTop != paddingBottom || paddingLeft != paddingRight);
        }
        public static void Padding(int inputLength, int kernelSize, int stride, PADDING_TYPE paddingType, CompatibilityModeEnum compatibilityMode, out int paddingStart, out int paddingEnd)
        {
            switch (paddingType)
            {
                case PADDING_TYPE.VALID:
                    paddingStart = paddingEnd = 0;
                    return;
                case PADDING_TYPE.SAME:
                    int outputLength = OutputLength(inputLength, kernelSize, stride, paddingType);
                    int totalPadding = Math.Max((outputLength - 1) * stride + kernelSize - inputLength, 0);
                    if (compatibilityMode == CompatibilityModeEnum.TensorFlow)
                    {
                        //see: https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
                        paddingStart = totalPadding / 2;
                        paddingEnd = totalPadding - paddingStart;
                    }
                    else
                    {
                        paddingStart = (totalPadding + 1) / 2;
                        paddingEnd = paddingStart;
                    }
                    return;
                case PADDING_TYPE.CAUSAL:
                    //see: https://github.com/keras-team/keras/issues/8751
                    paddingStart = kernelSize - 1;
                    paddingEnd = 0;
                    return;
                default:
                    throw new NotImplementedException("unknown padding type " + paddingType);
            }
        }


        #region PyTorch support
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            //see: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            var input_shape = PreviousLayers.Count == 0 ? new[] { -1, -1, -1,-1 } : PreviousLayers[0].OutputShape(666);
            int in_channels = input_shape[1];
            int out_channels = this.out_channels;


            var padding = "'"+_paddingType.ToString().ToLower()+"'";
            if (_paddingType == PADDING_TYPE.SAME && stride != 1)
            {
                padding = _isConv1D ? (kernel_width / 2).ToString() : ("(" + (kernel_height / 2) + "," + (kernel_width / 2) + ")");
            }
            if (_isConv1D)
            {
                Debug.Assert(!_isDepthwiseConvolution);
                constructorLines.Add("self." + LayerName + " = torch.nn.Conv1d(in_channels=" + in_channels + ", out_channels=" + out_channels + ", kernel_size="+kernel_width + ", stride=" + stride + ", padding=" + padding + ", bias=" + Utils.ToPython(bias) + ")");
            }
            else
            {
                int groups = 1;
                if (_isDepthwiseConvolution)
                {
                    groups = in_channels;
                    out_channels = in_channels * depth_multiplier;
                }
                constructorLines.Add("self." + LayerName + " = torch.nn.Conv2d(in_channels=" + in_channels + ", out_channels=" + out_channels + ", kernel_size=(" + kernel_height + "," + kernel_width + "), stride=" + stride + ", padding=" + padding + ", groups=" + groups + ", bias=" + Utils.ToPython(bias) + ")");
            }

            forwardLines.Add(GetPyTorchOutputVariableName() + " = self." + LayerName + "(" + GetInputVariableName() + ")");
            constructorLines.Add("torch.nn.init.xavier_normal_(self." + LayerName + ".weight)");
            if (bias)
            {
                constructorLines.Add("torch.nn.init.zeros_(self." + LayerName + ".bias)");
            }
        }

        #endregion
        private static int OutputLength(int inputLength, int kernelSize, int stride, PADDING_TYPE paddingType)
        {
            switch (paddingType)
            {
                case PADDING_TYPE.VALID:
                    return (inputLength - kernelSize) / stride + 1;
                case PADDING_TYPE.SAME:
                case PADDING_TYPE.CAUSAL:
                    return (inputLength - 1) / stride + 1;
                default:
                    throw new NotImplementedException("unknown padding type " + paddingType);
            }
        }
        private void Padding(int[] xShape4D, out int paddingTop, out int paddingBottom, out int paddingLeft, out int paddingRight)
        {
            Debug.Assert(xShape4D.Length == 4);

            var paddingTypeForHeight = _paddingType;
            var paddingTypeForWidth = _paddingType;
            if (paddingTypeForHeight == PADDING_TYPE.CAUSAL)
            {
                Debug.Assert(_isConv1D);
                paddingTypeForHeight = PADDING_TYPE.SAME;
            }
            Padding(xShape4D[2], kernel_height, stride, paddingTypeForHeight, Sample.CompatibilityMode, out paddingTop, out paddingBottom);
            Padding(xShape4D[3], kernel_width, stride, paddingTypeForWidth, Sample.CompatibilityMode, out paddingLeft, out paddingRight);
        }

        private int[] ConvolutionShape
        {
            get
            {
                var in_channels = PrevLayer.OutputShape(1)[1];
                return _isDepthwiseConvolution
                    ?new[] { depth_multiplier, in_channels, kernel_height, kernel_width }
                    :new[] { out_channels, in_channels, kernel_height, kernel_width };
            }
        }
        private bool bias => _convolutionBias != null;
        private int[] ConvolutionBiasShape
        {
            get
            {
                if (_isDepthwiseConvolution)
                {
                    var input_channels = PrevLayer.OutputShape(1)[1];
                    return new[] { depth_multiplier, input_channels, 1, 1 };
                }
                else
                {
                    return new[] { 1, out_channels, 1, 1 };
                }
            }
        }
        private GPUWrapper.ConvolutionAlgoPreference ConvolutionAlgoPreference => Sample.ConvolutionAlgoPreference;
        private static int[] PaddedXShape(int[] xShape, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            return new[] { xShape[0], xShape[1], paddingTop + xShape[2] + paddingBottom, paddingLeft + xShape[3] + paddingRight };
        }
    }
}
