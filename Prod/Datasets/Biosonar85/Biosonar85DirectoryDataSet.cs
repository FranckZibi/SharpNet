using SharpNet.CPU;
using System;
using System.Collections.Generic;
using SharpNet.Pictures;
using SharpNet.MathTools;
// ReSharper disable InconsistentlySynchronizedField

namespace SharpNet.Datasets.Biosonar85
{
    // ReSharper disable UnusedMember.Local

    public class Biosonar85DirectoryDataSet : DirectoryDataSet
    {
        private readonly Dictionary<int, BitmapContent> _cache = new(); 
        public Biosonar85DirectoryDataSet(List<List<string>> elementIdToPaths, List<int> elementIdToCategoryIndex, CpuTensor<float> expectedYIfAny, string name, Objective_enum objective, int channels, int numClass, List<Tuple<float, float>> meanAndVolatilityForEachChannel, ResizeStrategyEnum resizeStrategy, string[] featureNames, string[] y_IDs) 
            : base(elementIdToPaths, elementIdToCategoryIndex, expectedYIfAny, name, objective, channels, numClass, meanAndVolatilityForEachChannel, resizeStrategy, featureNames, y_IDs, "id")
        {
        }

        private readonly object _lockObject = new ();

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
        {
            if (xBuffer != null)
            {
                int channels = xBuffer.Shape[1];
                int targetHeight = xBuffer.Shape[2];
                int targetWidth = xBuffer.Shape[3];

                BitmapContent data = OriginalElementContent(elementId, channels, targetHeight, targetWidth, withDataAugmentation, isTraining);
                if (data == null)
                {
                    return;
                }

                Span<float> xBufferContent = xBuffer.RowSpanSlice(indexInBuffer, 1);
                int nextXBufferContentIndex = 0;
                for (int channel = 0; channel < data.GetChannels(); ++channel)
                {
                    var channelSpan = data.RowSpanSlice(channel, 1);
                    var acc = new DoubleAccumulator();
                    foreach(var b in channelSpan)
                    {
                        acc.Add(b);
                    }
                    var channelMean = (float)acc.Average;
                    var channelVolatility = (float)acc.Volatility;
                    for(int i=0;i<channelSpan.Length;++i)
                    {
                        xBufferContent[nextXBufferContentIndex++] = (channelSpan[i] - channelMean) / channelVolatility;
                    }
                }
            }

            if (yBuffer != null)
            {
                Y_DirectoryDataSet.CopyTo(Y_DirectoryDataSet.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        public override BitmapContent OriginalElementContent(int elementId, int channels, int targetHeight, int targetWidth, bool withDataAugmentation, bool isTraining)
        {
            try
            {
                if (_cache.ContainsKey(elementId))
                {
                    return _cache[elementId];
                }

                var tmp = BitmapContent.ValueFromSeveralSingleChannelBitmaps(_elementIdToPaths[elementId]);
                lock (_lockObject)
                {
                    if (!_cache.ContainsKey(elementId))
                    {
                        _cache[elementId] = tmp;
                    }
                }
                return _cache[elementId];
            }
            catch (Exception e)
            {
                Log.Error("Fail to load " + string.Join(" ", _elementIdToPaths[elementId]), e);
                return null;
            }
        }


    }
}