using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class MultiplyLayer : Layer
    {
        public MultiplyLayer(int previousLayerIndex1, int previousLayerIndex2, Network network, string layerName) : base(network, previousLayerIndex2, layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            Debug.Assert(previousLayerIndex1 >= 0);
            Debug.Assert(previousLayerIndex2 >= 0);
            //we add the identity shortcut connection
            AddPreviousLayer(previousLayerIndex1);

            if (!ValidLayerShapeToMultiply(PreviousLayer1.OutputShape(1), PreviousLayer2.OutputShape(1)))
            {
                throw new ArgumentException("invalid layers to multiply between " + PreviousLayer1 + " and " + previousLayerIndex2);
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            var x1 = allX[0];
            var x2 = allX[1]; //vector with the content of the diagonal matrix
            y.MultiplyTensor(x1, x2);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 2);
            var dx1 = allDx[0];
            var dx2 = allDx[1];
            var x1 = allX[0];
            var x2 = allX[1];
            Debug.Assert(dx1.SameShape(dy));
            Debug.Assert(dx1.SameShape(x1));
            Debug.Assert(dx2.SameShape(x2));


            StartBackwardTimer(Type() + ">SameShape");
            dx1.MultiplyTensor(dy, x2);
            StopBackwardTimer(Type() + ">SameShape");
            if (dx2.SameShape(dy))
            {
                StartBackwardTimer(Type() + ">SameShape");
                dx2.MultiplyTensor(dy, x1);
                StopBackwardTimer(Type() + ">SameShape");
            }
            else
            {
                StartBackwardTimer(Type() + ">DistinctShape");
                dx2.MultiplyEachRowIntoSingleValue(dy, x1);
                StopBackwardTimer(Type() + ">DistinctShape");
            }
        }
        #endregion

        #region serialization
        public MultiplyLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        #region layer clone
        public override Layer CloneForSlaveNetwork(Network newSlaveNetwork) { return new MultiplyLayer(this, newSlaveNetwork); }
        private MultiplyLayer(MultiplyLayer toCloneFromMasterNetwork, Network newSlaveNetwork) : base(toCloneFromMasterNetwork, newSlaveNetwork) { }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            var result1 = PreviousLayer1.OutputShape(batchSize);
            var result2 = PreviousLayer2.OutputShape(batchSize);
            var result = (int[])result1.Clone();
            result[1] = Math.Max(result1[1], result2[1]);
            return result;
        }

        private Layer PreviousLayer1 => PreviousLayers[0];
        private Layer PreviousLayer2 => PreviousLayers[1];
        //TODO add tests
        /// <summary>
        /// Check that the 2 layer shapes we want to multiply are valid:
        /// </summary>
        /// <param name="layerShape1"></param>
        /// <param name="layerShape2"></param>
        /// <returns></returns>
        private static bool ValidLayerShapeToMultiply(int[] layerShape1, int[] layerShape2)
        {
            if (layerShape1 == null
                || layerShape2 == null
                || layerShape1.Length != layerShape2.Length
                || layerShape1[0] != layerShape2[0] //must have same number of elements
                //|| layerShape1[1] != layerShape2[1] //must have same number of channels
            )
            {
                return false;
            }

            var layer1HasOnly1ForAllDimension = true;
            var layer2HasOnly1ForAllDimension = true;
            for (int i = 2; i < layerShape1.Length; ++i)
            {
                layer1HasOnly1ForAllDimension &= layerShape1[i] == 1;
                layer2HasOnly1ForAllDimension &= layerShape2[i] == 1;
            }

            if (layer1HasOnly1ForAllDimension || layer2HasOnly1ForAllDimension)
            {
                return true;
            }
            return layerShape1.SequenceEqual(layerShape2);
        }
    }
}