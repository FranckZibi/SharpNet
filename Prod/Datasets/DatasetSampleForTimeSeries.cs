using System;
using SharpNet.Networks;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Datasets;

public abstract class DatasetSampleForTimeSeries : AbstractDatasetSample
{
    public abstract int LoadEntry(TimeSeriesSinglePoint entry, float prev_Y, Span<float> xElementId, int idx, EncoderDecoder_NetworkSample networkSample, bool isEncoder);
    public abstract int GetInputSize(bool isEncoderInputSize);
    protected EncoderDecoder_NetworkSample encoderDecoder_NetworkSample { get; }


    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public bool Use_prev_Y = true;

    protected DatasetSampleForTimeSeries() {}
    protected DatasetSampleForTimeSeries(EncoderDecoder_NetworkSample encoderDecoderNetworkSample)
    {
        encoderDecoder_NetworkSample = encoderDecoderNetworkSample;
    }
}