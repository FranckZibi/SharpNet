using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Datasets.PascalVOC;
using SharpNet.Networks;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Datasets
{
    public class PascalVOCDataSet : DataSet
    {
        private readonly DirectoryDataSet _directoryDataSet;
        private readonly List<PascalVOCImageDescription> _annotations;

        public static readonly string[] _CategoryIndexToDescription = new[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        private PascalVOCDataSet(string vocDevKitDirectory, string subDirectory, List<Tuple<float, float>> meanAndVolatilityOfEachChannel, ResizeStrategyEnum resizeStrategy) 
            : base(subDirectory, 
                Objective_enum.Classification, 
                3, 
                meanAndVolatilityOfEachChannel, 
                resizeStrategy,
                new string[0],
                new string[0],
                new string[0],
                new string[0],
                true,
                ',')
        {
            var annotationsDirectory = Path.Combine(NetworkConfig.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "Annotations");
            var dataDirectory = Path.Combine(NetworkConfig.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "JPEGImages");
            var elementIdToCategoryIndex = new List<int>();
            var elementIdToDescription = new List<string>();
            var elementIdToPaths = new List<List<string>>();
            _annotations  = new List<PascalVOCImageDescription>();
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
                Log.Info(errorMsg);
            }

            for (int i = 0; i < allFiles.Length; ++i)
            {
                if (pascalVOCImageDescriptionsArray[i] == null)
                {
                    continue;
                }
                elementIdToPaths.Add(new List<string> { allFiles[i].FullName });
                elementIdToDescription.Add(Path.GetFileNameWithoutExtension(allFiles[i].Name));
                _annotations.Add(pascalVOCImageDescriptionsArray[i]);
                elementIdToCategoryIndex.Add(-1); //the category index of each object in the image can be found in the 'annotation' above
            }

            _directoryDataSet = new DirectoryDataSet(
                    elementIdToPaths, elementIdToDescription, elementIdToCategoryIndex, null
                    , subDirectory, Objective_enum.Classification, Channels, _CategoryIndexToDescription, meanAndVolatilityOfEachChannel, ResizeStrategyEnum.ResizeToTargetSize, null);
        }

        public static PascalVOCDataSet PascalVOC2007()
        {
            var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(114.31998f, 70.2124f), Tuple.Create(108.32131f, 69.44353f), Tuple.Create(99.885704f, 72.41173f) };
            return new PascalVOCDataSet("VOCdevkit2007", "VOC2007", meanAndVolatilityOfEachChannel, ResizeStrategyEnum.None);
        }
        public static PascalVOCDataSet PascalVOC2012()
        {
            var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(114.79079f, 70.890755f), Tuple.Create(109.19967f, 69.943886f), Tuple.Create(101.14191f, 72.790565f) };
            return new PascalVOCDataSet("VOCdevkit2012", "VOC2012", meanAndVolatilityOfEachChannel, ResizeStrategyEnum.None);
        }
    
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(xBuffer.Shape[0] == yBuffer?.Shape[0]);
            Debug.Assert(xBuffer.Shape[1] == Channels);
            throw new NotImplementedException();
        }

        public override int Count => _directoryDataSet.Count;

        public override int ElementIdToCategoryIndex(int elementId)
        {
            throw new ArgumentException("several categories may be associated with a single image");
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public int ElementIdToHeight(int elementId) { return _annotations[elementId].Height; }
        public int ElementIdToWidth(int elementId) { return _annotations[elementId].Width; }
        public override CpuTensor<float> Y => _directoryDataSet.Y;
    }

}
