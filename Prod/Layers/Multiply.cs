using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class Multiply : Layer
    {
        private static bool ShouldInvertInputMatrices(int mainMatrixLayerIndex, int diagonalMatrixLayerIndex, Network network)
        {
            return Utils.Product(network.Layers[diagonalMatrixLayerIndex].OutputShape(1)) > Utils.Product(network.Layers[mainMatrixLayerIndex].OutputShape(1));
        }


        public Multiply(int mainMatrixLayerIndex, int diagonalMatrixLayerIndex, Network network, string layerName) : base(network, ShouldInvertInputMatrices(mainMatrixLayerIndex, diagonalMatrixLayerIndex, network) ?new []{ diagonalMatrixLayerIndex , mainMatrixLayerIndex}: new[] { mainMatrixLayerIndex, diagonalMatrixLayerIndex },  layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            if (!ValidLayerShapeToMultiply(PreviousLayerMainMatrix.OutputShape(1), PreviousLayerDiagonalMatrix.OutputShape(1)))
            {
                throw new ArgumentException("invalid layers to multiply between " + PreviousLayerMainMatrix + " and " + diagonalMatrixLayerIndex);
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            var a = allX[0];
            var diagonalMatrix = allX[1]; //vector with the content of the diagonal matrix
            Debug.Assert(diagonalMatrix.Count <= a.Count);
            y.MultiplyTensor(a, diagonalMatrix);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 2);
            Debug.Assert(y_NotUsed == null);
            var dx1 = allDx[0];
            var dx2 = allDx[1];
            var a = allX[0];
            var diagonalMatrix = allX[1];
            Debug.Assert(diagonalMatrix.Count <= a.Count);
            Debug.Assert(dx1.SameShape(dy));
            Debug.Assert(dx1.SameShape(a));
            Debug.Assert(dx2.SameShape(diagonalMatrix));


            StartBackwardTimer(LayerType() + ">SameShape");
            dx1.MultiplyTensor(dy, diagonalMatrix);
            StopBackwardTimer(LayerType() + ">SameShape");
            if (dx2.SameShape(dy))
            {
                StartBackwardTimer(LayerType() + ">SameShape");
                dx2.MultiplyTensor(dy, a);
                StopBackwardTimer(LayerType() + ">SameShape");
            }
            else
            {
                StartBackwardTimer(LayerType() + ">DistinctShape");
                dx2.MultiplyEachRowIntoSingleValue(dy, a);
                StopBackwardTimer(LayerType() + ">DistinctShape");
            }
        }
        public override bool OutputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public static Multiply Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new Multiply(previousLayerIndexes[0], previousLayerIndexes[1], network, (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion




        #region PyTorch support
        //see : https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            forwardLines.Add(GetPyTorchOutputVariableName() + " = " + PreviousLayerMainMatrix.GetPyTorchOutputVariableName() + " * " + PreviousLayerDiagonalMatrix.GetPyTorchOutputVariableName());
        }

        #endregion

        public override int[] OutputShape(int batchSize)
        {
            var result1 = PreviousLayerMainMatrix.OutputShape(batchSize);
            var result2 = PreviousLayerDiagonalMatrix.OutputShape(batchSize);
            var result = (int[])result1.Clone();
            result[1] = Math.Max(result1[1], result2[1]);
            return result;
        }

        private Layer PreviousLayerMainMatrix => PreviousLayers[0];
        private Layer PreviousLayerDiagonalMatrix => PreviousLayers[1];
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