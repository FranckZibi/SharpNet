using SharpNet.CPU;
using System;
using System.Collections.Generic;
using SharpNet.Pictures;

namespace SharpNet.Datasets.Biosonar85
{
    // ReSharper disable UnusedMember.Local

    public class Biosonar85DirectoryDataSet : DirectoryDataSet
    {
        private Dictionary<int, BitmapContent> cache = new(); 
        public Biosonar85DirectoryDataSet(List<List<string>> elementIdToPaths, List<int> elementIdToCategoryIndex, CpuTensor<float> expectedYIfAny, string name, Objective_enum objective, int channels, int numClass, List<Tuple<float, float>> meanAndVolatilityForEachChannel, ResizeStrategyEnum resizeStrategy, string[] featureNames, string[] y_IDs) 
            : base(elementIdToPaths, elementIdToCategoryIndex, expectedYIfAny, name, objective, channels, numClass, meanAndVolatilityForEachChannel, resizeStrategy, featureNames, "id", y_IDs)
        {
        }

        private object lockObject = new object();

        public override BitmapContent OriginalElementContent(int elementId, int channels, int targetHeight, int targetWidth, bool withDataAugmentation, bool isTraining)
        {
            try
            {
                if (!cache.ContainsKey(elementId))
                {
                    var tmp = BitmapContent.ValueFromSeveralSingleChannelBitmaps(_elementIdToPaths[elementId]);
                    lock (lockObject)
                    {
                        if (!cache.ContainsKey(elementId))
                        {
                            cache[elementId] = tmp;
                        }
                    }
                }
                return cache[elementId];
            }
            catch (Exception e)
            {
                Log.Error("Fail to load " + string.Join(" ", _elementIdToPaths[elementId]), e);
                return null;
            }
        }


    }
}