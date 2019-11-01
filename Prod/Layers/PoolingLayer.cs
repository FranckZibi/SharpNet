﻿using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// x     (batchSize, x.C, x.H, x.W)
    ///
    /// y     (batchSize, x.C, y.H, y.W)
    ///          y.H = (x.H−poolingSize) / poolingStride + 1
    ///          y.W = (x.W−poolingSize) / poolingStride + 1
    /// </summary>
    public class PoolingLayer : Layer
    {
        #region Fields
        private readonly cudnnPoolingMode_t _poolingMode;
        private readonly int _poolingSize;
        private readonly int _poolingStride;
        public override Tensor y { get; protected set; }
        #endregion

        /// <summary>
        /// No need to configure the number of channels by filter: it is always the same as in previous layer
        /// </summary>
        /// <param name="poolingMode"></param>
        /// <param name="poolingSize"></param>
        /// <param name="poolingStride"></param>
        /// <param name="network"></param>
        public PoolingLayer(cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride, Network network) : base(network)
        {
            _poolingMode = poolingMode;
            _poolingSize = poolingSize;
            _poolingStride = poolingStride;
        }

        public override Layer Clone(Network newNetwork) { return new PoolingLayer(this, newNetwork); }
        private PoolingLayer(PoolingLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _poolingMode = toClone._poolingMode;
            _poolingSize = toClone._poolingSize;
            _poolingStride = toClone._poolingStride;
        }

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            x.Pooling(y, _poolingMode, _poolingSize, _poolingSize);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            var x = PrevLayer.y;
            dy.PoolingGradient(y, x, dx[0], _poolingMode, _poolingSize, _poolingStride);
        }

     
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (PoolingLayer)b;
            var equals = true;
            equals &= Utils.Equals(_poolingMode, other._poolingMode, id + ":_poolingMode", ref errors);
            equals &= Utils.Equals(_poolingSize, other._poolingSize, id + ":_poolingSize", ref errors);
            equals &= Utils.Equals(_poolingStride, other._poolingStride, id + ":_poolingStride", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_poolingSize), _poolingSize).Add(nameof(_poolingStride), _poolingStride)
                .Add(nameof(_poolingMode), (int)_poolingMode)
                .ToString();
        }
        public PoolingLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _poolingMode = (cudnnPoolingMode_t)(int)serialized[nameof(_poolingMode)];
            _poolingSize = (int)serialized[nameof(_poolingSize)];
            _poolingStride = (int)serialized[nameof(_poolingStride)];
        }
        #endregion
        public override string SummaryName()
        {
            return (IsMaxPooling(_poolingMode) ? "max_pooling2d_" : "average_pooling2d_") + (1 + NbLayerOfSameTypeBefore());
        }
        public override string Type() { return IsMaxPooling(_poolingMode) ? "MaxPooling" : "AveragePooling"; }
        public static bool IsMaxPooling(cudnnPoolingMode_t poolingMode)
        {
            return poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX ||
                   poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC;
        }
        public override string ToString()
        {
            var result = SummaryName()+": " + ShapeChangeDescription() + " size="+_poolingSize+" stride="+_poolingStride;
            result += " (" + MemoryDescription() + ")";
            return result;
        }
        public override int[] OutputShape(int batchSize)
        {
            var xShape = PrevLayer.OutputShape(batchSize);
            var yShape = PoolingOutputShape(xShape, _poolingSize, _poolingStride);
            Debug.Assert(yShape.Min() >= 1);
            return yShape;
        }

        /// <summary>
        /// Compute the pooling layer output shape given an input of shape 'shapeInput'
        /// </summary>
        /// <param name="shapeInput">(batchSize, x.C, heightInput, widthInput)</param>
        /// <param name="poolingSize">the pooling size is (poolingSize, poolingSize)</param>
        /// <param name="poolingStride">pooling stride</param>
        /// <returns>the output shape: (batchSize, x.C, y.H, y.W)</returns>
        public static int[] PoolingOutputShape(int[] shapeInput, int poolingSize, int poolingStride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(poolingStride >= 1);
            var batchSize = shapeInput[0];
            var heightInput = shapeInput[2];
            var widthInput = shapeInput[3];
            var heightOutput = (heightInput - poolingSize) / poolingStride + 1;
            var widthOutput = (widthInput - poolingSize) / poolingStride + 1;
            return new[] { batchSize, shapeInput[1], heightOutput, widthOutput };
        }
    }
}
