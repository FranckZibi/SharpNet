using System;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets.PlumeLabs88;

public class PlumeLabs88DirectoryDataSet : DataSet
{

    static PlumeLabs88DirectoryDataSet()
    {
        //var count = 1+ PlumeLabs88Utils.Shape_CHW[0] * PlumeLabs88Utils.Shape_CHW[1] * PlumeLabs88Utils.Shape_CHW[2];
    }

    private readonly PlumeLabs88DatasetSample _datasetSample;
    private readonly bool _isTrainingDataset;


   
    public override int[] YMiniBatch_Shape(int miniBatchSize)
    {
        return new[] { miniBatchSize, _datasetSample.NumClass };
    }
    public PlumeLabs88DirectoryDataSet(PlumeLabs88DatasetSample datasetSample,  bool isTrainingDataset)
        : base(PlumeLabs88Utils.NAME, datasetSample.GetObjective(), datasetSample.GetInputShapeOfSingleElement()[0], null, ResizeStrategyEnum.None, Array.Empty<string>(), datasetSample.CategoricalFeatures, datasetSample.IdColumns, isTrainingDataset, ',')
    {
        _datasetSample = datasetSample;
        _isTrainingDataset = isTrainingDataset;

        if (_isTrainingDataset)
        {
            var targetColumnContent = PlumeLabs88Utils.Load_YTrainPath().FloatColumnContent("TARGET");
            Y = CpuTensor<float>.New(targetColumnContent.Select(datasetSample.NormalizeTarget).ToArray(), _datasetSample.NumClass);
        }
        else
        {
            Y = null;
        }
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
        bool withDataAugmentation, bool isTraining)
    {
        var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
        Debug.Assert(xBufferSpan.Length == _datasetSample.FeatureByElement());
        _datasetSample.LoadElementIdIntoSpan(elementId, xBufferSpan, _isTrainingDataset);
        if (yBuffer != null && Y != null)
        {
            Y.RowSpanSlice(elementId, 1).CopyTo(yBuffer.RowSpanSlice(indexInBuffer, 1));
        }
    }

    public override int Count => 1+PlumeLabs88Utils.MaxId(_isTrainingDataset);
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }

    public override bool CanBeSavedInCSV => false;
    public override bool UseRowIndexAsId => true;

    public override CpuTensor<float> Y { get; }
}