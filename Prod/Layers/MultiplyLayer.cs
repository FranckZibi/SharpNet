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
        public override Tensor y { get; protected set; }

        public MultiplyLayer(int previousLayerIndex1, int previousLayerIndex2, Network network) : base(network, previousLayerIndex2)
        {
            Debug.Assert(LayerIndex >= 2);
            Debug.Assert(previousLayerIndex1 >= 0);
            Debug.Assert(previousLayerIndex2 >= 0);
            //we add the identity shortcut connection
            AddPreviousLayer(previousLayerIndex1);

            var result1 = PreviousLayer1.OutputShape(1);
            var result2 = PreviousLayer2.OutputShape(1);
            if (!ValidLayerShapeToMultiply(PreviousLayer1.OutputShape(1), PreviousLayer2.OutputShape(1)))
            {
                throw new ArgumentException("invalid layers to multiply between " + PreviousLayer1 + " and " + previousLayerIndex2);
            }
        }


        //TODO add tests
        /// <summary>
        /// Check that the 2 layer shapes we want to multiply are valid:
        /// </summary>
        /// <param name="layerShape1"></param>
        /// <param name="layerShape2"></param>
        /// <returns></returns>
        private static bool ValidLayerShapeToMultiply(int[] layerShape1, int[] layerShape2)
        {
            if (   layerShape1 == null 
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


        public override Layer Clone(Network newNetwork) { return new MultiplyLayer(this, newNetwork); }
        private MultiplyLayer(MultiplyLayer toClone, Network newNetwork) : base(toClone, newNetwork) { }

        #region serialization
        public MultiplyLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        private Layer PreviousLayer1 => PreviousLayers[0];
        private Layer PreviousLayer2 => PreviousLayers[1];
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var a = PreviousLayer1.y;
            var x = PreviousLayer2.y; //vector with the content of the diagonal matrix
            y.MultiplyTensor(a, x);
        }

        public override void BackwardPropagation(Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 2);
            Debug.Assert(y.SameShape(dy));
            Debug.Assert(allDx[0].SameShape(dy));
            Debug.Assert(allDx[0].SameShape(PreviousLayer1.y));
            Debug.Assert(allDx[1].SameShape(PreviousLayer2.y));
            allDx[0].MultiplyTensor(dy, PreviousLayer2.y);
            if (allDx[1].SameShape(dy))
            {
                allDx[1].MultiplyTensor(dy, PreviousLayer1.y);
            }
            else
            {
                allDx[1].MultiplyEachRowIntoSingleValue(dy, PreviousLayer1.y);
            }
        }

        public override int[] OutputShape(int batchSize)
        {
            var result1 = PreviousLayer1.OutputShape(batchSize);
            var result2 = PreviousLayer2.OutputShape(batchSize);
            var result = (int[])result1.Clone();
            result[1] = Math.Max(result1[1], result2[1]);
            return result;
        }
    }
}