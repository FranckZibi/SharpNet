using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// Invert second and third dimension of input 'x'
    /// Input 'x' shape:
    ///     [N, C, H) or [N, C, H,1]
    /// output shape:
    ///     [C, N, H] if AddOneDimensionInOutputShape is false
    ///     [C, N, H, 1] if AddOneDimensionInOutputShape is true
    /// </summary>
    public class SwitchSecondAndThirdDimensionLayer : Layer
    {
        private readonly bool AddOneDimensionInOutputShape;

        public SwitchSecondAndThirdDimensionLayer(bool addOneDimensionInOutputShape, Network network, string layerName) : base(network, layerName)
        {
            AddOneDimensionInOutputShape = addOneDimensionInOutputShape;
            Debug.Assert(LayerIndex >= 1);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(IsValidInputShape(allX[0].Shape));
            allX[0].SwitchSecondAndThirdDimension(y);
        }

        /// <summary>
        /// At this stage, we already know dy (output layer gradient)
        /// we want to compute the input layer gradient 'dx' by backward propagation
        /// </summary>
        /// <param name="y_NotUsed"></param>
        /// <param name="dy">already computed output layer gradient</param>
        /// <param name="dx">the value to compute</param>
        /// <param name="allX_NotUsed"></param>
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count == 1);
            Debug.Assert(IsValidInputShape(dx[0].Shape));
            dy.SwitchSecondAndThirdDimension(dx[0]);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(AddOneDimensionInOutputShape), AddOneDimensionInOutputShape)
                .ToString();
        }
        public static SwitchSecondAndThirdDimensionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var addOneDimensionInOutputShape = (bool)serialized[nameof(AddOneDimensionInOutputShape)];
            return new SwitchSecondAndThirdDimensionLayer(
                addOneDimensionInOutputShape,
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion


        private static bool IsValidInputShape(int[] shape)
        {
            return shape.Length == 3 || (shape.Length == 4 && shape[3] == 1);
        }

        public override int[] OutputShape(int batchSize)
        {
            var inputShape = PrevLayer.OutputShape(batchSize);
            Debug.Assert(IsValidInputShape(inputShape));

            int[] outputShape = new int[AddOneDimensionInOutputShape ? 4 : 3];
            outputShape[0] = inputShape[0];
            outputShape[2] = inputShape[1];
            outputShape[1] = inputShape[2];
            if (AddOneDimensionInOutputShape)
            {
                outputShape[3] = 1;
            }
            return outputShape;
        }
    }
}