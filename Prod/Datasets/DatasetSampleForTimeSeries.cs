using System;
using System.Collections.Generic;
using SharpNet.Networks;

namespace SharpNet.Datasets;

public abstract class DatasetSampleForTimeSeries : AbstractDatasetSample
{
    public abstract int LoadEntry(TimeSeriesSinglePoint entry, float prev_Y, Span<float> xElementId, int idx, EncoderDecoder_NetworkSample networkSample, bool isEncoder);
    public abstract int GetInputSize(bool isEncoderInputSize);
    public EncoderDecoder_NetworkSample encoderDecoder_NetworkSample { get; set; }


    public bool Use_prev_Y = true;

    protected DatasetSampleForTimeSeries() : base(new HashSet<string>())
    {
    }
}