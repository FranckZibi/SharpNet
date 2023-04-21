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
        /// maximum number of boxes to be selected by non-max suppression per class
        /// </summary>
        private readonly int _maxOutputSizePerClass;
        /// <summary>
        /// maximum number of boxes retained over all classes.
        /// </summary>
        private readonly int _maxOutputSize;
        /// <summary>
        /// the threshold for deciding whether boxes overlap too much with respect to IOU
        /// </summary>
        private readonly double _IOU_threshold;
        /// <summary>
        /// the threshold for deciding when to remove boxes based on score.
        /// </summary>
        private readonly double _score_Threshold;
        #endregion
        public NonMaxSuppressionLayer(int maxOutputSizePerClass, int maxOutputSize, double IOU_threshold, double score_Threshold, Network network, string layerName) : base(network, layerName)
        {
            _maxOutputSizePerClass = maxOutputSizePerClass;
            _maxOutputSize = maxOutputSize;
            _IOU_threshold = IOU_threshold;
            _score_Threshold = score_Threshold;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];


            //Non Max Suppression is implemented only on CPU side (no GPU)
            var yAsCpu = new CpuTensor<float>(y.Shape);
            x.CopyTo(yAsCpu);

            int predictionsByElement = x.Shape[1];
            for (int m = 0; m < x.Shape[0]; ++m)
            {
                var selectedPredictions = ExtractSelectedAfterNonMaxSuppression(yAsCpu, m, _maxOutputSizePerClass, _maxOutputSize, _IOU_threshold, _score_Threshold);

                //we set the box confidence of discarded predictions to 0
                var isSelected = new bool[predictionsByElement];
                selectedPredictions.ForEach(p => isSelected[p.IndexInInput] = true);
                for (int predictionId = 0; predictionId < predictionsByElement; ++predictionId)
                {
                    if (!isSelected[predictionId])
                    {
                        yAsCpu.Set(m, predictionId, 4, 0); 
                    }
                }
            }

            yAsCpu.CopyTo(y);
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
                .Add(nameof(_maxOutputSizePerClass), _maxOutputSizePerClass)
                .Add(nameof(_maxOutputSize), _maxOutputSize)
                .Add(nameof(_IOU_threshold), _IOU_threshold)
                .Add(nameof(_score_Threshold), _score_Threshold)
                .ToString();
        }
        public static NonMaxSuppressionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new NonMaxSuppressionLayer(
                (int)serialized[nameof(_maxOutputSizePerClass)],
                (int)serialized[nameof(_maxOutputSize)],
                (double)serialized[nameof(_IOU_threshold)], 
                (double)serialized[nameof(_score_Threshold)], 
                network, 
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public static List<PredictionDescription> ExtractSelectedAfterNonMaxSuppression(CpuTensor<float> tensor, int elementId, int maxOutputSizePerClass, int maxOutputSize, double IOU_threshold, double score_Threshold)
        {
            var predictions = PredictionsAboveBoxConfidenceThreshold(tensor, elementId, score_Threshold);
            var sortedPredictions = new List<PredictionDescription>(predictions.Where(p=>p.Score >= score_Threshold).OrderByDescending(p=>p.Score));
            if (sortedPredictions.Count == 0)
            {
                return new List<PredictionDescription>();
            }
            var countByClasses = new int[1+ sortedPredictions.Select(p=>p.ArgMaxClass).Max()];

            var selectedPredictions = new List<PredictionDescription>();
            for (int i = 0; i < sortedPredictions.Count; ++i)
            {
                if (sortedPredictions[i] == null || selectedPredictions.Count >= maxOutputSize || countByClasses[sortedPredictions[i].ArgMaxClass] >= maxOutputSizePerClass)
                {
                    continue;
                }
                selectedPredictions.Add(sortedPredictions[i]);
                ++countByClasses[sortedPredictions[i].ArgMaxClass];
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


        /// <summary>
        /// extract all predictions (related to image 'elementId') in the 'predictions' tensor with a box confidence above 'boxConfidence_Threshold'
        /// </summary>
        /// <param name="predictions">tensor with all predictions (for all images in the mini batch)</param>
        /// <param name="elementId">id of the image to process</param>
        /// <param name="boxConfidence_Threshold">minimum box confidence to select the associate prediction</param>
        /// <returns>list of predictions for image 'elementId' with a box confidence above the threshold</returns>
        private static List<PredictionDescription> PredictionsAboveBoxConfidenceThreshold(CpuTensor<float> predictions, int elementId, double boxConfidence_Threshold)
        {
            int predictionsByElement = predictions.Shape[1];
            int predictionLength = predictions.Shape[2];
            var predictionsAboveScoreThreshold = new List<PredictionDescription>();
            for (int predictionId = 0; predictionId < predictionsByElement; ++predictionId)
            {
                if (predictions.Get(elementId, predictionId, 4) >= boxConfidence_Threshold) //box confidence must be at least 'boxConfidence_Threshold'
                {
                    predictionsAboveScoreThreshold.Add(PredictionDescription.ValueOf(predictionId, predictions.ReadonlyContent.Slice(predictions.Idx(elementId, predictionId), predictionLength)));
                }
            }
            return predictionsAboveScoreThreshold;
        }

    }
}