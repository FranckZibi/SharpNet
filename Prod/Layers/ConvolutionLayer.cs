using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
        private readonly int _padding;
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

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ConvolutionLayer(bool isDepthwiseConvolution, int filtersCount, int depthMultiplier, int f, int stride, int padding, double lambdaL2Regularization, bool useBias, int previousLayerIndex, Network network, string layerName)
            : base(network, previousLayerIndex, layerName)
        {
            _isDepthwiseConvolution = isDepthwiseConvolution;
            _filtersCount = filtersCount;
            _depthMultiplier = depthMultiplier;
            if (_isDepthwiseConvolution && depthMultiplier != 1)
            {
                throw new NotImplementedException("only depthMultiplier=1 is supported");
            }
            _f = f;
            _stride = stride;
            _padding = padding;
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
            _padding = toClone._padding;
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
            equals &= Utils.Equals(_padding, other._padding, id + ":_padding", ref errors);
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
                .Add(nameof(_padding), _padding)
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
            _padding = (int)serialized[nameof(_padding)];
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
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x (conv) Convolution + ConvolutionBias
            x.Convolution(Convolution, _padding, _stride, y, _isDepthwiseConvolution);
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
            x.ConvolutionGradient(Convolution, dy, _padding, _stride, dx[0], ConvolutionGradients, _isDepthwiseConvolution);
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
            result += " padding=" + _padding + " stride=" + _stride;
            result += " Filter"+ Utils.ShapeToString(Convolution?.Shape);
            result += (UseBias)?" with Bias":" no Bias";
            result += " ("+ MemoryDescription()+")";
            return result;
        }

        public override int[] OutputShape(int batchSize)
        {
            var result =
                _isDepthwiseConvolution
                    ? DepthwiseConvolutionOutputShape(PrevLayer.OutputShape(batchSize), Convolution.Shape, _padding, _stride)
                    : StandardConvolutionOutputShape(PrevLayer.OutputShape(batchSize), Convolution.Shape, _padding, _stride);
            Debug.Assert(result.Min() >= 1);
            return result;
        }

        /// <summary>
        /// Compute the output shape fo a convolution layer, given input shape 'shapeInput' and convolution shape 'shapeConvolution'
        /// </summary>
        /// <param name="shapeInput">input shape: (batchSize, channelDepth, heightInput, widthInput)</param>
        /// <param name="shapeConvolution">convolution shape: (filtersCount, channelDepth, f, f)</param>
        /// <param name="padding"></param>
        /// <param name="stride"></param>
        /// <returns>output shape: (batchSize, filtersCount, heightOutput=H[heightInput], weightOutput=H[weightInput])</returns>
        public static int[] StandardConvolutionOutputShape(int[] shapeInput, int[] shapeConvolution, int padding, int stride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(shapeConvolution.Length == 4);
            Debug.Assert(padding >= 0);
            Debug.Assert(stride >= 1);
            Debug.Assert(shapeInput[1] == shapeConvolution[1]); //same channel depth for 'input shape' and 'convolution shape'
            Debug.Assert(shapeConvolution[2] == shapeConvolution[3]); //convolution height == convolution width
            var batchSize = shapeInput[0];
            var heightInput = shapeInput[2];
            var widthInput = shapeInput[3];
            var f = shapeConvolution[2];
            Debug.Assert(f % 2 == 1); // F must be odd
            var filtersCount = shapeConvolution[0];
            var heightOutput = (heightInput - f + 2 * padding) / stride + 1;
            var widthOutput = (widthInput - f + 2 * padding) / stride + 1;
            return new[] { batchSize, filtersCount, heightOutput, widthOutput };
        }

        /// <summary>
        /// Compute the output shape fo a convolution layer, given input shape 'shapeInput' and convolution shape 'shapeConvolution'
        /// </summary>
        /// <param name="shapeInput">input shape: (batchSize, channelDepth, heightInput, widthInput)</param>
        /// <param name="shapeConvolution">depthwise convolution shape: (1, channelDepth, f, f)</param>
        /// <param name="padding"></param>
        /// <param name="stride"></param>
        /// <returns>output shape: (batchSize, filtersCount, heightOutput=H[heightInput], weightOutput=H[weightInput])</returns>
        public static int[] DepthwiseConvolutionOutputShape(int[] shapeInput, int[] shapeConvolution, int padding, int stride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(shapeConvolution.Length == 4);
            Debug.Assert(padding >= 0);
            Debug.Assert(stride >= 1);
            Debug.Assert(1 == shapeConvolution[0]);             //filter count is always 1 for depthwise convolution
            Debug.Assert(shapeInput[1] == shapeConvolution[1]); //same channel depth for 'input shape' and 'convolution shape'
            Debug.Assert(shapeConvolution[2] == shapeConvolution[3]); //convolution height == convolution width
            var batchSize = shapeInput[0];
            var channelDepth = shapeInput[1];
            var heightInput = shapeInput[2];
            var widthInput = shapeInput[3];
            var f = shapeConvolution[2];
            Debug.Assert(f % 2 == 1); // F must be odd
            var heightOutput = (heightInput - f + 2 * padding) / stride + 1;
            var widthOutput = (widthInput - f + 2 * padding) / stride + 1;
            int depthMultiplier = shapeConvolution[0];
            return new[] { batchSize, channelDepth * depthMultiplier, heightOutput, widthOutput };
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
                    var channels = PrevLayer.OutputShape(1)[1];
                    return new[] { _depthMultiplier, channels, 1, 1 };
                }
                else
                {
                    return new[] { 1, _filtersCount, 1, 1 };
                }
            }
        }
    }
}