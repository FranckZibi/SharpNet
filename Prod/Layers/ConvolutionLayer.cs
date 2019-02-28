using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Optimizers;

namespace SharpNet
{
/*
  x                 (batchSize, x.C, x.H, x.W)
  Convolution       (FiltersCount, x.C, F, F)
  ConvolutionBias   (1, FiltersCount, 1, 1)
  y                 (batchSize, FiltersCount, H_output, W_output)
                        y.H = (x.H−F+2×pads) /Stride + 1
                        y.W = (x.W−F+2×pads) /Stride + 1
*/
    public class ConvolutionLayer : Layer
    {
        #region Private fields
        public override Tensor y { get; protected set; }        // (batchSize, FiltersCount, y.H, y.W)
        public override Tensor dy { get; protected set; }       // same as 'y'
        private int FiltersCount { get; }
        private readonly int _f;
        private readonly int _stride;
        private readonly int _padding;
        private readonly double _lambdaL2Regularization;
        public Tensor Convolution { get; }                    // (FiltersCount, x.C, F, F)
        public Tensor ConvolutionGradients { get; }            // same as 'Convolution'
        public Tensor ConvolutionBias { get; }                // (1, FiltersCount, 1, 1)
        public Tensor ConvolutionBiasGradients { get; }        // same as 'ConvolutionBias'
        private readonly Optimizer _optimizer;        //Adam optimization or SGD optimization or null
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ConvolutionLayer(int filtersCount, int f, int stride, int padding, double lambdaL2Regularization, int previousLayerIndex, Network network)
            : base(network, previousLayerIndex)
        {
            this.FiltersCount = filtersCount;
            this._f = f;
            this._stride = stride;
            this._padding = padding;
            _lambdaL2Regularization = lambdaL2Regularization;
            Convolution = Initialize_Convolution();
            ConvolutionGradients = Network.NewTensor(Convolution.Shape, nameof(ConvolutionGradients));
            var convolutionBiasShape = new[] { 1, FiltersCount, 1, 1 };
            ConvolutionBias = Network.NewTensor(convolutionBiasShape, nameof(ConvolutionBias));
            ConvolutionBiasGradients = Network.NewTensor(ConvolutionBias.Shape, nameof(ConvolutionBiasGradients));
            _optimizer = Network.GetOptimizer(Convolution.Shape, ConvolutionBias.Shape);
        }

        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(FiltersCount), FiltersCount).Add(nameof(_f), _f).Add(nameof(_stride), _stride).Add(nameof(_padding), _padding)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(Convolution).Add(ConvolutionGradients).Add(ConvolutionBias).Add(ConvolutionBiasGradients)
                .Add(_optimizer?.Serialize())
                .ToString();
        }
        public static ConvolutionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new ConvolutionLayer(serialized, network);
        }

        private ConvolutionLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            FiltersCount = (int)serialized[nameof(FiltersCount)];
            _f = (int)serialized[nameof(_f)];
            _stride = (int)serialized[nameof(_stride)];
            _padding = (int)serialized[nameof(_padding)];
            _lambdaL2Regularization = (int)serialized[nameof(_lambdaL2Regularization)];
            Convolution = (Tensor)serialized[nameof(Convolution)];
            ConvolutionGradients = (Tensor)serialized[nameof(ConvolutionGradients)];
            ConvolutionBias = (Tensor)serialized[nameof(ConvolutionBias)];
            ConvolutionBiasGradients = (Tensor)serialized[nameof(ConvolutionBiasGradients)];
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x (conv) Convolution + ConvolutionBias
            x.Convolution(Convolution, _padding, _stride, y);
            ConvolutionBias.BroadcastConvolutionBiasToOutput(y);
        }

        // dy => ConvolutionGradient & dx
        public override void BackwardPropagation()
        {
            //At this stage, we already know dy, we want to compute dx by backward propagation
            Debug.Assert(y.SameShape(dy));
            Debug.Assert(ConvolutionBiasGradients.SameShape(ConvolutionBias));

            //we update dy if necessary (shortcut connection to a futur layer)
            Update_dy_With_GradientFromShortcutIdentityConnection(); 

            // we compute _convolutionBiasGradient
            dy.ConvolutionBackwardBias(ConvolutionBiasGradients);

            // we compute ConvolutionGradient (& dx if PrevLayer is not the input layer)
            var x = PrevLayer.y;
            var dx = PrevLayer.dy;
            if (PrevLayer.NextLayers.Count>=2 && PrevLayer.NextLayers[1].LayerIndex == LayerIndex)
            {
                //shortcut identity connection: conv layer was used only to change dimension
                dx = PrevLayer.dyIdentityConnection;
                Debug.Assert(dx != null);
            }

            x.ConvolutionGradient(Convolution, dy, _padding, _stride, dx, ConvolutionGradients);

            if (UseL2Regularization)
            {
                var batchSize = y.Shape[0];
                var alpha = _lambdaL2Regularization / batchSize;
                ConvolutionGradients.Update_Adding_Alpha_X(alpha, Convolution);
            }
        }
        public override void UpdateWeights(double learningRate)
        {
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Convolution, ConvolutionGradients, ConvolutionBias, ConvolutionBiasGradients);
        }
        public override int TotalParams
        {
            get
            {
                if (Convolution == null)
                {
                    return 0;
                }
                return Convolution.Count + ConvolutionBias.Count;
            }
        }
        public override void Dispose()
        {
            base.Dispose();
            _optimizer?.Dispose();
        }
        public override string SummaryName() {return "Conv2D";}
        public override string ToString()
        {
            var result = SummaryName()+": " + ShapeChangeDescription();
            result += " padding=" + _padding + " stride=" + _stride;
            result += " Filter"+ Utils.ShapeToString(Convolution?.Shape);
            result += " ("+ MemoryDescription()+")";
            return result;
        }
        public override int[] OutputShape(int batchSize)
        {
            var result = Tensor.ConvolutionOutputShape(PrevLayer.OutputShape(batchSize), Convolution.Shape, _padding, _stride);
            Debug.Assert(result.Min() >= 1);
            return result;
        }

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;
        public override List<Tensor> TensorsIndependantOfBatchSize
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
        private Tensor Initialize_Convolution()
        {
            var fanIn = ChannelCountByFilter * _f * _f;
            var fanOut = FiltersCount;
            var stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));
            var convolutionShape = new[] { FiltersCount, ChannelCountByFilter, _f, _f };
            return Network.RandomMatrixNormalDistribution(convolutionShape, 0.0 /* mean */, stdDev, nameof(Convolution));
        }
        private int ChannelCountByFilter => PrevLayer.OutputShape(1)[1];

    }
}