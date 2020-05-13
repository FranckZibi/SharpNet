using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class NonMaxSuppressionLayer : Layer
    {
        #region private fields
        /// <summary>
        /// the threshold for deciding when to remove boxes based on score.
        /// </summary>
        private readonly float _score_Threshold;
        /// <summary>
        /// the threshold for deciding whether boxes overlap too much with respect to IOU
        /// </summary>
        private readonly float _IOU_threshold;
        private readonly int _maxOutputSize;
        private readonly int _maxOutputSizePerClass;
        #endregion

        public NonMaxSuppressionLayer(float score_Threshold, float IOU_threshold, int maxOutputSize, int maxOutputSizePerClass, Network network, string layerName) : base(network, layerName)
        {
            _score_Threshold = score_Threshold;
            _IOU_threshold = IOU_threshold;
            _maxOutputSize = maxOutputSize;
            _maxOutputSizePerClass = maxOutputSizePerClass;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];

            var xAsCpu = new CpuTensor<float>(x.Shape);
            x.CopyTo(xAsCpu);
            int predictionsByElement = x.Shape[1];
            int predictionLength = x.Shape[2];
            for (int m = 0; m < x.Shape[0]; ++m)
            {
                var element = xAsCpu.ElementSlice(m).AsFloatCpuSpan;
                var elementPredictions = new List<PredictionDescription>();
                for (int predictionId = 0; predictionId < predictionsByElement; ++predictionId)
                {
                    elementPredictions.Add(PredictionDescription.ValueOf(predictionId, element.Slice(predictionId*predictionLength,predictionLength)));
                }
                var selectedPredictions = ExtractSelectedAfterNonMaxSuppression(elementPredictions, _score_Threshold, _IOU_threshold, _maxOutputSize, _maxOutputSizePerClass);
                var isSelected = new bool[predictionsByElement];
                selectedPredictions.ForEach(p=> isSelected[p.IndexInInput] = true);
                for (int predictionId = 0; predictionId < predictionsByElement; ++predictionId)
                {
                    if (!isSelected[predictionId])
                    {
                        xAsCpu.Set(m, predictionId, 4, 0); //we set the box confidence of removed element to 0
                    }
                }
            }
            xAsCpu.CopyTo(x);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_score_Threshold), _score_Threshold)
                .Add(nameof(_IOU_threshold), _IOU_threshold)
                .Add(nameof(_maxOutputSize), _maxOutputSize)
                .Add(nameof(_maxOutputSizePerClass), _maxOutputSizePerClass)
                .ToString();
        }
        public static NonMaxSuppressionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new NonMaxSuppressionLayer(
                (float)serialized[nameof(_score_Threshold)],
                (float)serialized[nameof(_IOU_threshold)],
                (int)serialized[nameof(_maxOutputSize)],
                (int)serialized[nameof(_maxOutputSizePerClass)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            return PrevLayer.OutputShape(batchSize);
        }

        private static List<PredictionDescription> ExtractSelectedAfterNonMaxSuppression(IReadOnlyList<PredictionDescription> predictions, float score_Threshold, float IOU_threshold, int maxOutputSize, int maxOutputSizePerClass)
        {
            var selectedPredictions = new List<PredictionDescription>();
            var sortedPredictions = new List<PredictionDescription>(predictions.Where(p=>p.Score >= score_Threshold).OrderByDescending(p=>p.Score));

            int[] countByCategories = new int[1+ sortedPredictions.Select(p=>p.ArgMaxClass).Max()];

            for (int i = 0; i < sortedPredictions.Count; ++i)
            {
                if (sortedPredictions[i] == null || selectedPredictions.Count >= maxOutputSize || countByCategories[sortedPredictions[i].ArgMaxClass] >= maxOutputSizePerClass)
                {
                    continue;
                }
                ++countByCategories[sortedPredictions[i].ArgMaxClass];
                for (int j = i + 1; j < sortedPredictions.Count; ++j)
                {
                    if (   sortedPredictions[j] != null
                        && sortedPredictions[i].ArgMaxClass == sortedPredictions[j].ArgMaxClass
                        && sortedPredictions[i].Box.IoU(sortedPredictions[j].Box) >= IOU_threshold)
                    {
                        sortedPredictions[j] = null;
                    }
                }
            }
            return selectedPredictions;
        }
    }
}