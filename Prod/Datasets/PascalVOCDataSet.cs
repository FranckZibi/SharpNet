using System;
using System.Collections.Generic;
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

        public static readonly string[] CategoryIndexToDescription = new[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        private PascalVOCDataSet(string vocDevKitDirectory, string subDirectory, List<Tuple<float, float>> meanAndVolatilityOfEachChannel, ResizeStrategyEnum resizeStrategy) 
            : base(subDirectory, 
                Objective_enum.Classification, 
                meanAndVolatilityOfEachChannel, 
                resizeStrategy,
                new string[0],
                new string[0],
                "",
                null, //TODO
                ',')
        {
            var annotationsDirectory = Path.Combine(NetworkSample.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "Annotations");
            var dataDirectory = Path.Combine(NetworkSample.DefaultDataDirectory, vocDevKitDirectory, subDirectory, "JPEGImages");
            var elementIdToCategoryIndex = new List<int>();
            var elementIdToId = new List<string>();
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
                elementIdToId.Add(Path.GetFileNameWithoutExtension(allFiles[i].Name));
                _annotations.Add(pascalVOCImageDescriptionsArray[i]);
                elementIdToCategoryIndex.Add(-1); //the category index of each object in the image can be found in the 'annotation' above
            }

            _directoryDataSet = new DirectoryDataSet(
                    elementIdToPaths, elementIdToCategoryIndex, null
                    , subDirectory, Objective_enum.Classification, 3, CategoryIndexToDescription, meanAndVolatilityOfEachChannel, ResizeStrategyEnum.ResizeToTargetSize, null, elementIdToId.ToArray());
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
    
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
        {
            throw new NotImplementedException();
        }
        public override int[] Y_Shape()
        {
            return new[] { Count, CategoryIndexToDescription.Length };
        }
        public override int Count => _directoryDataSet.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            throw new ArgumentException("several categories may be associated with a single image");
        }
        public int ElementIdToHeight(int elementId) { return _annotations[elementId].Height; }
        public int ElementIdToWidth(int elementId) { return _annotations[elementId].Width; }
    }
}
