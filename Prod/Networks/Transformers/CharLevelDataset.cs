using System;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;

namespace SharpNet.Networks.Transformers;

public class CharLevelDataset : DataSet
{
    private readonly CharLevelTransformersDatasetSample _datasetSample;
    private readonly CpuTensor<float> _yCharLevelDataset;
    [NotNull] private readonly int[] _textToSequence;

    public CharLevelDataset(
        CharLevelTransformersDatasetSample datasetSample,
        string name,
        string text,
        [NotNull] Tokenizer tokenizer)
        : base(name,
            datasetSample.GetObjective(),
            null,
            ResizeStrategyEnum.None,
            new string[0],
            new string[0],
            "",
            null,
            ',')
    {
        _datasetSample = datasetSample;
        _textToSequence = tokenizer.TextsToSequences(new[]{text}).SelectMany(v => v).ToArray();
        _yCharLevelDataset = null;
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
        var yBufferSpan = (yBuffer==null)?null:yBuffer.RowSpanSlice(indexInBuffer, 1);
        Debug.Assert(xBufferSpan.Length == _datasetSample.max_length);
        Debug.Assert(yBufferSpan == null || yBufferSpan.Length == _datasetSample.max_length);
        for (int j = 0; j < _datasetSample.max_length; ++j)
        {
            xBufferSpan[j] = _textToSequence[elementId+j];
            if (yBuffer != null)
            {
                yBufferSpan[j] = _textToSequence[elementId+j+1];
            }
        }
    }
    
    public override int Count => _textToSequence.Length - _datasetSample.max_length - 1;
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }

    public override CpuTensor<float> Y => _yCharLevelDataset;
}