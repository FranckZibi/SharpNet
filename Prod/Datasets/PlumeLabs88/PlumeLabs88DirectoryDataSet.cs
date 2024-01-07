using System;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets.PlumeLabs88;

public class PlumeLabs88DirectoryDataSet : DataSet

{
    private readonly bool _isTrainingDataset;
    [CanBeNull] private readonly CpuTensor<float> _yPlumeLabs88DirectoryDataSet;

    public PlumeLabs88DirectoryDataSet(PlumeLabs88DatasetSample datasetSample,  bool isTrainingDataset)
        : base(PlumeLabs88Utils.NAME, datasetSample, null, ResizeStrategyEnum.None, Array.Empty<string>(), datasetSample.RowInTargetFormatPredictionToID(isTrainingDataset), ',')
    {
        _isTrainingDataset = isTrainingDataset;

        if (_isTrainingDataset)
        {
            var targetColumnContent = PlumeLabs88Utils.Load_YTrainPath().FloatColumnContent("TARGET");
            _yPlumeLabs88DirectoryDataSet = CpuTensor<float>.New(targetColumnContent.Select(datasetSample.NormalizeTarget).ToArray(), datasetSample.NumClass);
        }
        else
        {
            _yPlumeLabs88DirectoryDataSet = null;
        }
    }

    private PlumeLabs88DatasetSample PlumeLabs88DatasetSample => (PlumeLabs88DatasetSample)DatasetSample;

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
        bool withDataAugmentation, bool isTraining)
    {
        if (xBuffer != null)
        {
            var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
            //we ensure that we have the good number of features by element
            Debug.Assert(xBufferSpan.Length == Utils.Product(DatasetSample.X_Shape(1)));
            PlumeLabs88DatasetSample.LoadElementIdIntoSpan(elementId, xBufferSpan, _isTrainingDataset);
        }
        if (yBuffer != null && _yPlumeLabs88DirectoryDataSet != null)
        {
            _yPlumeLabs88DirectoryDataSet.RowSpanSlice(elementId, 1).CopyTo(yBuffer.RowSpanSlice(indexInBuffer, 1));
        }
    }

    public override CpuTensor<float> LoadFullY()
    {
        return _yPlumeLabs88DirectoryDataSet;
    }


    public override int Count => 1+ PlumeLabs88DatasetSample.DatasetMaxId(_isTrainingDataset);
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }
    public override bool CanBeSavedInCSV => false;
    public override bool UseRowIndexAsId => true;
}
