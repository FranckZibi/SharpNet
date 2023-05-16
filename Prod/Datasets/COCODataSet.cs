using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    // ReSharper disable once ClassNeverInstantiated.Global
    public class COCODataSet : DataSet
    {
        #region private fields
        // ReSharper disable once NotAccessedField.Local
        private readonly string _cocoDirectory;
        #endregion

        #region constructor
        public COCODataSet(string cocoDirectory, List<Tuple<float, float>> meanAndVolatilityOfEachChannel, ResizeStrategyEnum resizeStrategy) : base("COCO", 
            Objective_enum.Classification, 
            3, 
            meanAndVolatilityOfEachChannel, 
            resizeStrategy,
            new string[0],
            new string[0],
            "",
            null, //TODO
            ',')
        {
            _cocoDirectory = cocoDirectory;
            throw new NotImplementedException();
        }
        #endregion

        public static readonly string[] CategoryIndexToDescription = new[]
        {
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };


        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
        {
            throw new NotImplementedException();
        }

        public override int Count => throw new NotImplementedException();

        public override int ElementIdToCategoryIndex(int elementId)
        {
            throw new ArgumentException("several categories may be associated with a single image");
        }
        public override CpuTensor<float> Y => throw new NotImplementedException();
    }
}