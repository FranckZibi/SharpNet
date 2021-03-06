﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public sealed class SplittedFileDataSet : AbstractDataSet
    {
        #region private fields
        [NotNull] private readonly int[] _elementIdToCategoryIndex;
        [NotNull] private readonly List<string> Files;
        [NotNull] private readonly int[] FirstElementIdInFile;
        [NotNull] private readonly int[] LastElementIdInFile;
        [NotNull] private readonly int[] _singleElementShape_CHW; /* shape in format (channels, height, width) */
        [NotNull] private readonly Func<byte, int> CategoryByteToCategoryIndex;
        #endregion

        public SplittedFileDataSet([NotNull] List<string> files, string name, [NotNull] string[] categoryDescriptions, [NotNull] int[] singleElementShape_CHW, [CanBeNull]  List<Tuple<float, float>> meanAndVolatilityForEachChannel, [NotNull] Func<byte, int> categoryByteToCategoryIndex)
            : base(name, singleElementShape_CHW[0], categoryDescriptions, meanAndVolatilityForEachChannel, ResizeStrategyEnum.None)
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

            //TODO : initialize Y tensor
            Y = new CpuTensor<float>(new[] { Count, categoryDescriptions.Length });
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            Debug.Assert(Channels == xBuffer.Shape[1]); //same number of channels

            var targetHeight = xBuffer.Shape[2];
            var targetWidth = xBuffer.Shape[3];

            //we load the element content from the file
            var xByte = LoadElementIdFromFile(elementId);
            var xBufferContent = xBuffer.SpanContent;

            //we initialize 'xBuffer'
            int xByteIndex = 1;
            int xBufferIndex = xBuffer.Idx(indexInBuffer);
            for (int channel = 0; channel < Channels; ++channel)
            {
                for (int row = 0; row < targetHeight; ++row)
                {
                    for (int col = 0; col < targetWidth; ++col)
                    {
                        var val = (double)xByte[xByteIndex++];
                        val = (val - OriginalChannelMean(channel)) / OriginalChannelVolatility(channel);
                        xBufferContent[xBufferIndex++] = (float)val;
                    }
                }
            }

            //we initialize 'yBuffer'
            var categoryIndex = CategoryByteToCategoryIndex(xByte[0]);
            _elementIdToCategoryIndex[elementId] = categoryIndex;
            for (int cat = 0; cat < Y.Shape[1]; ++cat)
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
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }
        public override int Count { get; }
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