﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Datasets.PascalVOC;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public class PascalVOCDataSet : AbstractDataSet
    {
        private readonly DirectoryDataSet _directoryDataSet;
        private readonly List<PascalVOCImageDescription> _annotations = new List<PascalVOCImageDescription>();

        public static readonly string[] CategoryIndexToDescription = new[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        public PascalVOCDataSet(string vocDevKitDirectory, string subDirectory, List<Tuple<float, float>> meanAndVolatilityOfEachChannel, Logger logger) : base(subDirectory, 3, CategoryIndexToDescription.Length, meanAndVolatilityOfEachChannel, logger)
        {
            var annotationsDirection = Path.Combine(NetworkConfig.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "Annotations");
            var dataDirectory = Path.Combine(NetworkConfig.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "JPEGImages");
            _directoryDataSet = new DirectoryDataSet(annotationsDirection, dataDirectory, logger, subDirectory, Channels, -1, -1, CategoryIndexToDescription, meanAndVolatilityOfEachChannel, false,
                (a, b, c, d, e, f) => DefaultCompute_CategoryIndex_Description_FullName(a, b, c, d, e, f, _annotations));
        }

        public static PascalVOCDataSet PascalVOC2007(Logger logger)
        {
            var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(114.31998f, 70.2124f), Tuple.Create(108.32131f, 69.44353f), Tuple.Create(99.885704f, 72.41173f) };
            return new PascalVOCDataSet("VOCdevkit2007", "VOC2007", meanAndVolatilityOfEachChannel, logger);
        }
        public static PascalVOCDataSet PascalVOC2012(Logger logger)
        {
            var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(114.79079f, 70.890755f), Tuple.Create(109.19967f, 69.943886f), Tuple.Create(101.14191f, 72.790565f) };
            return new PascalVOCDataSet("VOCdevkit2012", "VOC2012", meanAndVolatilityOfEachChannel, logger);
        }


        private static void DefaultCompute_CategoryIndex_Description_FullName(
            string annotationsDirectory,
            string dataDirectory,
            Logger logger,
            List<int> elementIdToCategoryIndex,
            List<string> elementIdToDescription,
            List<List<string>> elementIdToSubPath,
            List<PascalVOCImageDescription> pascalVOCImageDescriptions
            )
        {
            elementIdToCategoryIndex.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();
            pascalVOCImageDescriptions.Clear();
            var missingAnnotations = new List<string>();

            var allFiles = new DirectoryInfo(dataDirectory).GetFiles().OrderBy(f => f.Name).ToArray();
            var pascalVOCImageDescriptionsArray = new PascalVOCImageDescription[allFiles.Length];
            void ProcessSingleElement(int elementIndex)
            {
                var description = Path.GetFileNameWithoutExtension(allFiles[elementIndex].Name);
                var annotationPath = Path.Combine(annotationsDirectory, description + ".xml");
                if (File.Exists(annotationPath))
                {
                    pascalVOCImageDescriptionsArray[elementIndex] = PascalVOCImageDescription.ValueOf(annotationPath);
                }
                else
                {
                    lock (missingAnnotations)
                    {
                        missingAnnotations.Add(annotationPath);
                    }
                }
            }
            Parallel.For(0, allFiles.Length, ProcessSingleElement);

            if (missingAnnotations.Any())
            {
                var errorMsg = missingAnnotations.Count+ " file(s) with missing annotation (ignoring all of them) " +Environment.NewLine + "First 5:"+Environment.NewLine + string.Join(Environment.NewLine, missingAnnotations.Take(5));
                logger.Info(errorMsg);
            }

            for (int i = 0; i < allFiles.Length; ++i)
            {
                if (pascalVOCImageDescriptionsArray[i] == null)
                {
                    continue;
                }
                var subPath = allFiles[i].Name;
                elementIdToSubPath.Add(new List<string> { subPath });
                elementIdToDescription.Add(Path.GetFileNameWithoutExtension(subPath));
                pascalVOCImageDescriptions.Add(pascalVOCImageDescriptionsArray[i]);
                elementIdToCategoryIndex.Add(-1); //the category index of each object in the image can be found in the 'annotation' above
            }
        }

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            Debug.Assert(xBuffer.Shape[0] == yBuffer.Shape[0]);
            Debug.Assert(xBuffer.Shape[1] == Channels);
            var targetHeight = xBuffer.Shape[2];
            var targetWidth = xBuffer.Shape[3];
            throw new NotImplementedException();
        }

        public override int Count => _directoryDataSet.Count;

        public override int ElementIdToCategoryIndex(int elementId)
        {
            throw new ArgumentException("several categories may be associated with a single image");
        }


        /// <summary>
        /// in Pascal VOC dataSet, each element may have different height
        /// </summary>
        public override int Height => -1;
        public int ElementIdToHeight(int elementId) { return _annotations[elementId].Height; }
        /// <summary>
        /// in Pascal VOC dataSet, each element may have different width
        /// </summary>
        public override int Width => -1;
        public int ElementIdToWidth(int elementId) { return _annotations[elementId].Width; }
        public override CpuTensor<float> Y { get; }
    }

}
