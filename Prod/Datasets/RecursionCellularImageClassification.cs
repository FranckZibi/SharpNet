
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpNet.Datasets
{
    public static class RecursionCellularImageClassification
    {
        private const int Channels = 3;
        private const int Categories = 4;

        public static DirectoryDataSet ValueOf(string csvFilename, string trainingSetDirectory, int height, int width, Logger logger)
        { 
            return new DirectoryDataSet(
                csvFilename,
                trainingSetDirectory,
                logger,
                "RecursionCellularImageClassification",
                Channels,
                height,
                width,
                AbstractDataSet.DefaultGetCategoryIndexToDescription(Categories),
                false, //we do not ignore zero pixels
                Compute_CategoryIndex_Description_FullName);
        }

        private static void Compute_CategoryIndex_Description_FullName(
              string csvFileName,
              string directoryWithElements,
              List<int> elementIdToCategoryIndex,
              List<string> elementIdToDescription,
              List<List<string>> elementIdToSubPath,
              Logger _logger)
        {
            elementIdToCategoryIndex.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();
            var lines = File.ReadAllLines(csvFileName);
            //we skip the first line : it is the header (id_code,experiment,plate,well,sirna)
            for (var index = 1; index < lines.Length; index++)
            {
                //ex of line: HEPG2-01_1_B03,HEPG2-01,1,B03,513
                var lineContent = lines[index].Split(',').ToArray();
                var id_code = lineContent[0];        //description, ex: HEPG2-01_1_B03
                var experiment = lineContent[1];     //the cell type and batch number, ex: HEPG2-01
                var plate = lineContent[2];          //plate number within the experiment, ex: 1
                var well = lineContent[3];           //location on the plate, ex B03
                int sirna = lineContent.Length >= 5 ? int.Parse(lineContent[4]) : -1; //the target, ex: 513
                var subPathPart = Path.Combine(experiment, "Plate" + plate);
                for (int s = 1; s <= 2; ++s)
                {
                    var subPaths = new List<string>();
                    for (int w = 1; w <= 6; ++w)
                    {
                        //ex of subPath: HEPG2-01\Plate1\B03_s1_w1.png"
                        var subPath = Path.Combine(subPathPart, well+"_s"+s+"_w"+w+".png");
                        var filename = Path.Combine(directoryWithElements, subPath);
                        if (File.Exists(filename))
                        {
                            subPaths.Add(subPath);
                        }
                    }
                    if (subPaths.Count != Channels)
                    {
                        _logger.Info("WARNING: can't find well " + well + "/ s=" + s + " in directory " + subPathPart);
                    }
                    else
                    {
                        //elementIdToCategoryIndex.Add(sirna);
                        elementIdToCategoryIndex.Add(sirna % 4);
                        elementIdToDescription.Add(id_code);
                        elementIdToSubPath.Add(subPaths);
                    }
                }
            }
        }
    }
}
