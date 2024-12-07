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
    [NotNull] private readonly int[] _textToSequence;

    public TransformerDataset(
        TransformerDatasetSample datasetSample,
        string name,
        string text,
        [NotNull] Tokenizer tokenizer)
        : base(name,
            datasetSample,
            //datasetSample.GetObjective(),
            null,
            ResizeStrategyEnum.None,
            new string[0]
            //datasetSample.IsCategoricalColumn
            )
    {
        _textToSequence = tokenizer.TextsToSequences(new[]{text}).SelectMany(v => v).ToArray();
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        if (xBuffer != null)
        {
            var xBufferSpan = xBuffer.RowSpanSlice(indexInBuffer, 1);
            Debug.Assert(xBufferSpan.Length == TransformerDatasetSample.max_length);
            for (int j = 0; j < TransformerDatasetSample.max_length; ++j)
            {
                xBufferSpan[j] = _textToSequence[elementId+j];
            }
        }

        if (yBuffer != null)
        {
            var yBufferSpan = yBuffer.RowSpanSlice(indexInBuffer, 1);
            Debug.Assert(yBufferSpan.Length == TransformerDatasetSample.vocab_size);
            yBufferSpan.Clear();
            yBufferSpan[_textToSequence[elementId + TransformerDatasetSample.max_length]] = 1;
            /*
            Debug.Assert(yBufferSpan.Length == TransformerDatasetSample.max_length * TransformerDatasetSample.vocab_size);
            yBufferSpan.Clear();
            for (int j = 0; j< TransformerDatasetSample.max_length; ++j)
            {
                Debug.Assert( (elementId + j + 1) < _textToSequence.Length);
                var new_token_index_in_vocab_size = _textToSequence[elementId+j+1];
                yBufferSpan[j*TransformerDatasetSample.vocab_size + new_token_index_in_vocab_size] = 1;
            }
            */
        }
    }

    private TransformerDatasetSample TransformerDatasetSample => (TransformerDatasetSample)DatasetSample;
    public override int Count => _textToSequence.Length - TransformerDatasetSample.max_length - 1;
    public override int ElementIdToCategoryIndex(int elementId)
    {
        throw new NotImplementedException();
    }
}
