﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class SplittedFileDataSet : AbstractDataSet
    {
        #region private fields
        [NotNull] private readonly int[] _elementIdToCategoryIndex;
        [NotNull] private readonly List<string> Files;
        [NotNull] private readonly int[] FirstElementIdInFile;
        [NotNull] private readonly int[] LastElementIdInFile;
        [NotNull] private readonly int[] _singleElementShape_CHW; /* shape in format (channels, height, width) */
        [NotNull] private readonly Func<byte, int> CategoryByteToCategoryIndex;
        #endregion

        public SplittedFileDataSet([NotNull] List<string> files, string name, int categories, [NotNull] int[] singleElementShape_CHW, [CanBeNull]  List<Tuple<float, float>> meanAndVolatilityForEachChannel, [NotNull] Func<byte, int> categoryByteToCategoryIndex)
            : base(name, singleElementShape_CHW[0], categories, meanAndVolatilityForEachChannel, null)
        {
            //Currently only pictures (channels x height x width) are supported
            Debug.Assert(singleElementShape_CHW.Length == 3);
            Files = files;
            _singleElementShape_CHW = singleElementShape_CHW;
            //+1 byte to store the category associated with each element
            long bytesInSingleElement = Utils.Product(singleElementShape_CHW) + 1;
            FirstElementIdInFile = new int[Files.Count];
            LastElementIdInFile = new int[Files.Count];
            for (var index = 0; index < Files.Count; index++)
            {
                FirstElementIdInFile[index] = index == 0 ? 0 : (LastElementIdInFile[index - 1]+1);
                int elementsInFile = (int) (Utils.FileLength(Files[index]) / bytesInSingleElement);
                LastElementIdInFile[index] = FirstElementIdInFile[index] + elementsInFile-1;
            }

            CategoryByteToCategoryIndex = categoryByteToCategoryIndex;
            Count = LastElementIdInFile.Last()+1;

            // ReSharper disable once VirtualMemberCallInConstructor
            _elementIdToCategoryIndex = new int[Count];
            for (int i = 0; i < _elementIdToCategoryIndex.Length; ++i)
            {
                _elementIdToCategoryIndex[i] = -1;
            }

            Y = new CpuTensor<float>(Y_Shape, null, "Y");
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            Debug.Assert(Channels == xBuffer.Shape[1]); //same number of channels
            Debug.Assert(Height == xBuffer.Shape[2]); //same height
            Debug.Assert(Width == xBuffer.Shape[3]); //same width

            //we load the element content from the file
            var xByte = LoadElementIdFromFile(elementId);

            //we initialize 'xBuffer'
            int xByteIndex = 1;
            int xBufferIndex = xBuffer.Idx(indexInBuffer);
            for (int channel = 0; channel < Channels; ++channel)
            {
                for (int row = 0; row < Height; ++row)
                {
                    for (int col = 0; col < Width; ++col)
                    {
                        var val = (double)xByte[xByteIndex++];
                        val = (val - OriginalChannelMean(channel)) / OriginalChannelVolatility(channel);
                        xBuffer.Content[xBufferIndex++] = (float)val;
                    }
                }
            }

            //we initialize 'yBuffer'
            var categoryIndex = CategoryByteToCategoryIndex(xByte[0]);
            _elementIdToCategoryIndex[elementId] = categoryIndex;
            for (int cat = 0; cat < Categories; ++cat)
            {
                yBuffer?.Set(indexInBuffer, cat, (cat == categoryIndex) ? 1f : 0f);
                Y.Set(elementId, cat, (cat == categoryIndex) ? 1f : 0f);
            }
        }
        public override int ElementIdToCategoryIndex(int elementId)
        {
            var res = _elementIdToCategoryIndex[elementId];
            Debug.Assert(res>=0);
            return res;
        }
        //public List<Tuple<float, float>> ComputeMeanAndVolatilityForEachChannel()
        //{
        //    var sumSumSquareCountForEachChannel = new float[Channels * 3]; //3 == sum + sumSquare + count
        //    int nbPerformed = 0;
        //    int firstElementIdInFile = 0;
        //    foreach (var file in Files)
        //    {
        //        LoadTensorsContainingElementId(firstElementIdInFile);
        //        Parallel.For(firstElementIdInFile, firstElementIdInFile+NbElementsInFile(file), elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(elementId, sumSumSquareCountForEachChannel, false, ref nbPerformed));
        //        firstElementIdInFile += NbElementsInFile(file);
        //    }
        //    return Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(sumSumSquareCountForEachChannel);
        //}
        public override int Count { get; }
        public override int Height => _singleElementShape_CHW[1];
        public override int Width => _singleElementShape_CHW[1];
        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return "X => " + Y;
        }
        public static List<string> AllBinFilesInDirectory(string directory, params string[] filesPrefix)
        {
            var result = new List<string>();
            foreach (var filePrefix in filesPrefix)
            {
                var fileName = Path.Combine(directory, filePrefix + ".bin");
                if (File.Exists(fileName))
                {
                    result.Add(fileName);
                }

                for (int i = 1;; ++i)
                {
                    fileName = Path.Combine(directory, filePrefix + "_" + i + ".bin");
                    if (!File.Exists(fileName))
                    {
                        break;
                    }
                    result.Add(fileName);
                }
            }
            return result;
        }

        private byte[] LoadElementIdFromFile(int elementId)
        {
            var fileIndex = Array.BinarySearch(LastElementIdInFile, elementId);
            if (fileIndex < 0)
            {
                fileIndex = ~fileIndex;
            }
            Debug.Assert(elementId >= FirstElementIdInFile[fileIndex]);
            Debug.Assert(elementId <= LastElementIdInFile[fileIndex]);

            var bytesInSingleElement = (Utils.Product(_singleElementShape_CHW) + 1);
            int positionInFile = (elementId - FirstElementIdInFile[fileIndex]) * bytesInSingleElement;
            return Utils.ReadPartOfFile(Files[fileIndex], positionInFile, bytesInSingleElement);
        }
    }
}