using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    public sealed class DepthwiseConvolutionLayer : Layer
    {
        #region Private fields
        public override Tensor y { get; protected set; }        // (batchSize, _depthMultiplier*x.C, y.H, y.W)
        private readonly int _f;
        private readonly int _stride;
        private readonly int _padding;
        private readonly int _depthMultiplier;
        private readonly double _lambdaL2Regularization;
        private readonly Optimizer _optimizer;                  //Adam optimization or SGD optimization or null
        private bool UseBias => DepthwiseConvolutionBias != null;
        public Tensor DepthwiseConvolution { get; }                      // (_depthMultiplier, x.C, F, F)
        public Tensor DepthwiseConvolutionGradients { get; }            // same as 'DepthwiseConvolution'
        public Tensor DepthwiseConvolutionBias { get; private set; }    // (_depthMultiplier, x.C, 1, 1) or null is no bias should be used
        public Tensor DepthwiseConvolutionBiasGradients { get; private set; }        // same as 'DepthwiseConvolutionBias'  or null is no bias should be used
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public DepthwiseConvolutionLayer(int f, int stride, int padding, int depthMultiplier, double lambdaL2Regularization, bool useBias, int previousLayerIndex, Network network, string layerName)
            : base(network, previousLayerIndex, layerName)
        {
            _f = f;
            _stride = stride;
            _padding = padding;
            _depthMultiplier = depthMultiplier;
            if (depthMultiplier != 1)
            {
                throw new NotImplementedException("only depthMultiplier=1 is supported");
            }

            _lambdaL2Regularization = lambdaL2Regularization;
            DepthwiseConvolution = Network.NewNotInitializedTensor(DepthwiseConvolutionShape, nameof(DepthwiseConvolution));
            DepthwiseConvolutionGradients = Network.NewNotInitializedTensor(DepthwiseConvolution.Shape, nameof(DepthwiseConvolutionGradients));
            if (useBias)
            {
                DepthwiseConvolutionBias = Network.NewNotInitializedTensor(DepthwiseConvolutionBiasShape, nameof(DepthwiseConvolutionBias));
                DepthwiseConvolutionBiasGradients = Network.NewNotInitializedTensor(DepthwiseConvolutionBias.Shape, nameof(DepthwiseConvolutionBiasGradients));
            }
            _optimizer = Network.GetOptimizer(DepthwiseConvolution.Shape, DepthwiseConvolutionBias?.Shape);
            ResetWeights(false);
        }

        public override Layer Clone(Network newNetwork) { return new DepthwiseConvolutionLayer(this, newNetwork); }
        private DepthwiseConvolutionLayer(DepthwiseConvolutionLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _f = toClone._f;
            _stride = toClone._stride;
            _padding = toClone._padding;
            _depthMultiplier = toClone._depthMultiplier;
            _lambdaL2Regularization = toClone._lambdaL2Regularization;
            DepthwiseConvolution = toClone.DepthwiseConvolution?.Clone(newNetwork.GpuWrapper);
            DepthwiseConvolutionGradients = toClone.DepthwiseConvolutionGradients?.Clone(newNetwork.GpuWrapper);
            DepthwiseConvolutionBias = toClone.DepthwiseConvolutionBias?.Clone(newNetwork.GpuWrapper);
            DepthwiseConvolutionBiasGradients = toClone.DepthwiseConvolutionBiasGradients?.Clone(newNetwork.GpuWrapper);
            _optimizer = toClone._optimizer?.Clone(newNetwork);
        }

        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (DepthwiseConvolutionLayer)b;
            var equals = true;
            equals &= Utils.Equals(_f, other._f, id + ":_f", ref errors);
            equals &= Utils.Equals(_stride, other._stride, id + ":_stride", ref errors);
            equals &= Utils.Equals(_padding, other._padding, id + ":_padding", ref errors);
            equals &= Utils.Equals(_depthMultiplier, other._depthMultiplier, id + ":_depthMultiplier", ref errors);
            equals &= Utils.Equals(_lambdaL2Regularization, other._lambdaL2Regularization, epsilon, id + ":_lambdaL2Regularization", ref errors);
            equals &= _optimizer.Equals(other._optimizer, epsilon, id + ":Optimizer", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_f), _f).Add(nameof(_stride), _stride)
                .Add(nameof(_padding), _padding)
                .Add(nameof(_depthMultiplier), _depthMultiplier)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(DepthwiseConvolution).Add(DepthwiseConvolutionGradients)
                .Add(DepthwiseConvolutionBias).Add(DepthwiseConvolutionBiasGradients)
                .Add(_optimizer.Serialize())
                .ToString();
        }
        public DepthwiseConvolutionLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _f = (int)serialized[nameof(_f)];
            _stride = (int)serialized[nameof(_stride)];
            _padding = (int)serialized[nameof(_padding)];
            _depthMultiplier = (int)serialized[nameof(_depthMultiplier)];
            _lambdaL2Regularization = (double)serialized[nameof(_lambdaL2Regularization)];
            DepthwiseConvolution = (Tensor)serialized[nameof(DepthwiseConvolution)];
            DepthwiseConvolutionGradients = (Tensor)serialized[nameof(DepthwiseConvolutionGradients)];
            //Bias may be null if it has been disabled
            DepthwiseConvolutionBias = serialized.TryGet<Tensor>(nameof(DepthwiseConvolutionBias));
            DepthwiseConvolutionBiasGradients = serialized.TryGet<Tensor>(nameof(DepthwiseConvolutionBiasGradients));
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }
        #endregion
        public override int DisableBias()
        {
            int nbDisabledWeights = (DepthwiseConvolutionBias?.Count ?? 0) + (DepthwiseConvolutionBiasGradients?.Count ?? 0);
            DepthwiseConvolutionBias?.Dispose();
            DepthwiseConvolutionBias = null;
            DepthwiseConvolutionBiasGradients?.Dispose();
            DepthwiseConvolutionBiasGradients = null;
            return nbDisabledWeights;
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x (conv) Convolution + ConvolutionBias
            x.Convolution(DepthwiseConvolution, _padding, _stride, y, true);
            if (UseBias)
            {
                DepthwiseConvolutionBias.BroadcastConvolutionBiasToOutput(y);
            }
        }
        // dy => ConvolutionGradient & dx
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            Debug.Assert(DepthwiseConvolutionBias == null || DepthwiseConvolutionBias.SameShape(DepthwiseConvolutionBiasGradients));

            // we compute ConvolutionBiasGradients
            if (UseBias)
            {
                dy.ConvolutionBackwardBias(DepthwiseConvolutionBiasGradients);
            }

            // we compute ConvolutionGradient (& dx if PrevLayer is not the input layer)
            var x = PrevLayer.y;
            x.ConvolutionGradient(DepthwiseConvolution, dy, _padding, _stride, dx[0], DepthwiseConvolutionGradients, true);
            if (UseL2Regularization)
            {
                var batchSize = y.Shape[0];
                var alpha = 2 * batchSize * (float)_lambdaL2Regularization;
                DepthwiseConvolutionGradients.Update_Adding_Alpha_X(alpha, DepthwiseConvolution);
            }
        }
        public override void UpdateWeights(double learningRate)
        {
            if (!Trainable)
            {
                return;
            }
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, DepthwiseConvolution, DepthwiseConvolutionGradients, DepthwiseConvolutionBias, DepthwiseConvolutionBiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            var fanIn = DepthwiseConvolution.MultDim0;
            var fanOut = DepthwiseConvolution.Shape[0];
            var stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));
            DepthwiseConvolution.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, stdDev);
            DepthwiseConvolutionGradients.ZeroMemory();
            DepthwiseConvolutionBias?.ZeroMemory();
            DepthwiseConvolutionBiasGradients?.ZeroMemory();
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override int TotalParams => DepthwiseConvolution.Count + (DepthwiseConvolutionBias?.Count ?? 0);

        public override void Dispose()
        {
            base.Dispose();
            _optimizer?.Dispose();
        }
        public override string Type() { return "DepthwiseConv2D"; }

        public override string ToString()
        {
            var result = LayerName + ": " + ShapeChangeDescription();
            result += " padding=" + _padding + " stride=" + _stride;
            result += " Filter" + Utils.ShapeToString(DepthwiseConvolution?.Shape);
            result += (UseBias) ? " with Bias" : " no Bias";
            result += " (" + MemoryDescription() + ")";
            return result;
        }
        public override int[] OutputShape(int batchSize)
        {
            var result = DepthwiseConvolutionOutputShape(PrevLayer.OutputShape(batchSize), DepthwiseConvolution.Shape, _padding, _stride);
            Debug.Assert(result.Min() >= 1);
            return result;
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
            return new[] { batchSize, channelDepth* depthMultiplier, heightOutput, widthOutput };
        }

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;
        public override List<Tensor> TensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { DepthwiseConvolution, DepthwiseConvolutionGradients, DepthwiseConvolutionBias, DepthwiseConvolutionBiasGradients };
                if (_optimizer != null)
                {
                    result.AddRange(_optimizer.EmbeddedTensors);
                }
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        private int[] DepthwiseConvolutionShape
        {
            get
            {
                var channels = PrevLayer.OutputShape(1)[1];
                return new[] {_depthMultiplier, channels, _f, _f };

            }
        }
        private int[] DepthwiseConvolutionBiasShape
        {
            get
            {
                var channels = PrevLayer.OutputShape(1)[1];
                return new[] { _depthMultiplier, channels, 1, 1};
           }
        }
    }
}