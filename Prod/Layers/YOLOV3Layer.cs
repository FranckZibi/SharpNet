using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.ObjectDetection;

namespace SharpNet.Layers
{
    /// input shape (n, totalPrediction, predictionDescription)
    /// predictionDescription:
    ///   predictionDescription[0] & [1]        = box center (colCenter + rowCenter)
    ///   predictionDescription[2] & [3]        = box size (width + height)
    ///   predictionDescription[4]              = box confidence
    ///   predictionDescription[5 and more]     = for each class class confidence
    /// </summary>

    public class PredictionDescription
    {
       

        public int IndexInInput { get; }
        public BoundingBox Box { get; }
        private float BoxConfidence { get; }
        public int ArgMaxClass  { get; }
        private float ArgMaxClassPrediction  { get; }

        public float Score => BoxConfidence* ArgMaxClassPrediction;


        public string CaptionFor(string[] categoryIndexToDescription)
        {
            return categoryIndexToDescription[ArgMaxClass]+" "+Math.Round(Score, 4);
        }

        public PredictionDescription(int indexInInput, BoundingBox box, float boxConfidence, int argMaxClass, float argMaxClassPrediction)
        {
            IndexInInput = indexInInput;
            Box = box;
            BoxConfidence = boxConfidence;
            ArgMaxClass = argMaxClass;
            ArgMaxClassPrediction = argMaxClassPrediction;
        }

        public static PredictionDescription ValueOf(int indexInInput, ReadOnlySpan<float> data)
        {
            int argMax = 5;
            for (int i = 6; i < data.Length; ++i)
            {
                if (data[i] > data[argMax])
                {
                    argMax = i;
                }
            }
            var box = new BoundingBox(data[0], data[1], data[2], data[3]);
            return new PredictionDescription(indexInInput, box, data[4], argMax - 5, data[argMax] );
        }


    }


    /// <summary>
    /// input shape (n, c, h , w)
    ///     with c = AnchorCount* ( 2 (for box centers) + 2 (for box size) + 1 (for box confidence) +_categories (for each class)
    /// output shape (n, AnchorCount*h*w, c/AnchorCount) 
    /// </summary>
    public class YOLOV3Layer : Layer
    {
        #region private fields
        private readonly int[] _anchors;
        #endregion

        public YOLOV3Layer(int[] anchors, int previousLayerIndex, Network network, string layerName) : base(network, new[] { previousLayerIndex}, layerName)
        {
            Debug.Assert(anchors.Length%2 == 0);
            _anchors = anchors;
            Debug.Assert(AnchorCount == 3);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            var inputImageShape = Layers[0].OutputShape(1);
            y.YOLOV3Forward(x, inputImageShape[2], inputImageShape[3], _anchors);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_anchors), _anchors).ToString();
        }
        public static YOLOV3Layer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new YOLOV3Layer(
                (int[])serialized[nameof(_anchors)],
                previousLayerIndexes[0],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        /// <summary>
        /// input shape is:     (n, c, h, w)
        /// output shape is:    (n, AnchorCount*h*w, c/AnchorCount) 
        /// </summary>
        public override int[] OutputShape(int batchSize)
        {
            var prevLayerOutputShape = PrevLayer.OutputShape(batchSize);
            Debug.Assert(prevLayerOutputShape[1] % AnchorCount == 0);
            var result = new int[3];
            result[0] = prevLayerOutputShape[0];
            result[1] = AnchorCount * prevLayerOutputShape[2] * prevLayerOutputShape[3];
            result[2] = prevLayerOutputShape[1] / AnchorCount;
            return result;
        }

        private int AnchorCount => _anchors.Length / 2;
    }
}