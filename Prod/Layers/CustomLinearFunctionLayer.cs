using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNet.Layers
{

    public interface ILayerNeedingDataSetForForwardPropagation
    { 
        void ForwardPropagationWithDataSet(List<Tensor> allX, Tensor y, bool isTraining, IDataSet dataSet, Memory<int> batchIndexToElementIdInDataSet);
    }

    /// <summary>
    /// Layer that computes:
    ///     y[batchIndex] = Beta[batchIndex] * x[batchIndex] + Alpha[batchIndex]
    /// </summary>
    public class CustomLinearFunctionLayer : Layer, ILayerNeedingDataSetForForwardPropagation
    {
        private readonly float BetaConstant;

        /// <summary>
        /// tensor of shape (batchSize) where:
        ///     Beta[batchIndex] : the slope of the linear function to use for element at index 'batchId'
        /// </summary>
        [NotNull] private Tensor Beta;
        /// <summary>
        /// tensor of shape (batchSize) where:
        ///     Alpha[batchIndex] : the constant to add in the linear function for element at index 'batchId'
        /// </summary>
        [NotNull] private Tensor Alpha;

        public CustomLinearFunctionLayer(float betaConstant, Network network, string layerName = "") : base(network, layerName)
        {
            BetaConstant = betaConstant;
            Beta = GetFloatTensor(new []{1,1});
            Alpha = GetFloatTensor(Beta.Shape);
        }
        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            throw new ArgumentException("should never be called, call "+nameof(ForwardPropagationWithDataSet)+" instead");
        }

        public void ForwardPropagationWithDataSet(List<Tensor> allX, Tensor y, bool isTraining, IDataSet dataSet, Memory<int> batchIndexToElementIdInDataSet)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            var batchSize = y.Shape[0];

            //x and y must be of shape: (batchSize, 1)
            Debug.Assert(x.SameShape(y));
            Debug.Assert(x.SameShape(new [] {batchSize, 1}));

            Debug.Assert(batchIndexToElementIdInDataSet.Length == batchSize);

            var dataSetWithExpectedAverage = dataSet as IDataSetWithExpectedAverage;
            if (dataSetWithExpectedAverage == null)
            {
                throw new ArgumentException("IDataSet must implement " + nameof(IDataSetWithExpectedAverage) + " but received " + dataSet);
            }

            var betaInCpu= new float[batchSize];
            var alphaInCpu= new float[batchSize];
            int idx = 0;
            foreach (var elementId in batchIndexToElementIdInDataSet.Span)
            {
                betaInCpu[idx] = BetaConstant; //todo: use custom value for alpha
                alphaInCpu[idx] = dataSetWithExpectedAverage.ElementIdToExpectedAverage(elementId);
                ++idx;
            }

            GetFloatTensor(ref Beta, x.Shape);
            new CpuTensor<float>(Beta.Shape, betaInCpu).CopyTo(Beta);

            GetFloatTensor(ref Alpha, x.Shape);
            new CpuTensor<float>(Alpha.Shape, alphaInCpu).CopyTo(Alpha);

            y.LinearFunction(Beta, x, Alpha);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 1);
            Debug.Assert(allDx[0].SameShape(dy));
            allDx[0].LinearFunction(Beta, dy, null);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
        {
            var result = new List<Tensor> { Beta, Alpha };
            result.RemoveAll(t => t == null);
            return result;
        }

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(BetaConstant), BetaConstant)
                .ToString();
        }
        public static CustomLinearFunctionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new CustomLinearFunctionLayer(
                (float)serialized[nameof(BetaConstant)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
    }
}