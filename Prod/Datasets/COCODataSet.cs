using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class COCODataSet : AbstractDataSet
    {
        #region private fields
        private readonly string _cocoDirectory;
        #endregion

        #region cosntructor
        public COCODataSet(string cocoDirectory, List<Tuple<float, float>> meanAndVolatilityOfEachChannel) : base("COCO", 3, CategoryIndexToDescription.Length, meanAndVolatilityOfEachChannel)
        {
            Y = null;
            _cocoDirectory = cocoDirectory;
            throw new NotImplementedException();
        }
        #endregion

        public static readonly string[] CategoryIndexToDescription = new[]
        {
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };


        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            throw new NotImplementedException();
        }

        public override int Count => throw new NotImplementedException();

        public override int ElementIdToCategoryIndex(int elementId)
        {
            throw new ArgumentException("several categories may be associated with a single image");
        }


        /// <summary>
        /// in COCO dataSet, each element may have different height
        /// </summary>
        public override int Height => -1;
        /// <summary>
        /// in COCO dataSet, each element may have different width
        /// </summary>
        public override int Width => -1;
        public override CpuTensor<float> Y { get; }
    }
}