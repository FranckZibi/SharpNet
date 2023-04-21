﻿using System;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets.PlumeLabs88;

public class PlumeLabs88DirectoryDataSet : DataSet
{
    private readonly PlumeLabs88DatasetSample _datasetSample;
    private readonly bool _isTrainingDataset;
    private readonly CpuTensor<float> _yPlumeLabs88DirectoryDataSet;

    public PlumeLabs88DirectoryDataSet(PlumeLabs88DatasetSample datasetSample,  bool isTrainingDataset)
        : base(PlumeLabs88Utils.NAME, datasetSample.GetObjective(), datasetSample.GetInputShapeOfSingleElement()[0], null, ResizeStrategyEnum.None, Array.Empty<string>(), datasetSample.CategoricalFeatures, datasetSample.IdColumn, datasetSample.RowInTargetFormatPredictionToID(isTrainingDataset), isTrainingDataset, ',')
    {
        _datasetSample = datasetSample;
        _isTrainingDataset = isTrainingDataset;

        if (_isTrainingDataset)
        {
            var targetColumnContent = PlumeLabs88Utils.Load_YTrainPath().FloatColumnContent("TARGET");
            _yPlumeLabs88DirectoryDataSet = CpuTensor<float>.New(targetColumnContent.Select(datasetSample.NormalizeTarget).ToArray(), _datasetSample.NumClass);
        }
        else
        {
            _yPlumeLabs88DirectoryDataSet = null;
        }
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
        bool withDataAugmentation, bool isTraining)
    {
        var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
        Debug.Assert(xBufferSpan.Length == _datasetSample.FeatureByElement());
        _datasetSample.LoadElementIdIntoSpan(elementId, xBufferSpan, _isTrainingDataset);
        if (yBuffer != null && _yPlumeLabs88DirectoryDataSet != null)
        {
            _yPlumeLabs88DirectoryDataSet.RowSpanSlice(elementId, 1).CopyTo(yBuffer.RowSpanSlice(indexInBuffer, 1));
        }
    }

    public override int Count => 1+_datasetSample.DatasetMaxId(_isTrainingDataset);
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }

    public override bool CanBeSavedInCSV => false;
    public override bool UseRowIndexAsId => true;

    public override CpuTensor<float> Y => _yPlumeLabs88DirectoryDataSet;
}