using System;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;

namespace SharpNet.Networks.Transformers;

public class TransformerDataset : DataSet
{
    private readonly TransformerDatasetSample _datasetSample;
    [NotNull] private readonly int[] _textToSequence;

    public TransformerDataset(
        TransformerDatasetSample datasetSample,
        string name,
        string text,
        [NotNull] Tokenizer tokenizer)
        : base(name,
            datasetSample.GetObjective(),
            null,
            ResizeStrategyEnum.None,
            new string[0],
            new string[0])
    {
        _datasetSample = datasetSample;
        _textToSequence = tokenizer.TextsToSequences(new[]{text}).SelectMany(v => v).ToArray();
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        if (xBuffer != null)
        {
            var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
            Debug.Assert(xBufferSpan.Length == _datasetSample.max_length);
            for (int j = 0; j < _datasetSample.max_length; ++j)
            {
                xBufferSpan[j] = _textToSequence[elementId+j];
            }
        }
        if (yBuffer != null)
        {
            var yBufferSpan = yBuffer.RowSpanSlice(indexInBuffer, 1);
            Debug.Assert(yBufferSpan.Length == _datasetSample.vocab_size);
            yBufferSpan.Clear();
            yBufferSpan[_textToSequence[elementId + _datasetSample.max_length]] = 1;
        }
    }

    public override int Count => _textToSequence.Length - _datasetSample.max_length - 1;
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }

    public override AbstractDatasetSample GetDatasetSample() => _datasetSample;
}
