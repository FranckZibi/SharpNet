using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public static class DogsVsCats
    {
        public static DirectoryDataSet ValueOf(string directory, int height, int width, Logger logger)
        {
            return new DirectoryDataSet(
                "",
                directory,
                logger,
                "DogsVsCats", //Name
                3, //Channels
                height,
                width,
                new[] {"cat", "dog"},
                false, //we do not ignore zero pixels
                ComputeCategoryIndexDescriptionFullName);
        }

        private static void ComputeCategoryIndexDescriptionFullName(
            string csvFileName,
            string dataDirectory,
            List<int> elementIdToCategoryIndex,
            List<string> elementIdToDescription,
            List<List<string>> elementIdToSubPath,
            Logger logger)
        {
            elementIdToCategoryIndex.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();

            foreach (var fileinfo in new DirectoryInfo(dataDirectory).GetFiles().Where(x=>PictureTools.IsPicture(x.FullName)))
            {
                var fullName = fileinfo.FullName;
                var description = Path.GetFileNameWithoutExtension(fullName);
                elementIdToDescription.Add(description);
                int categoryIndex = description.StartsWith("dog") ? 1 : 0;
                elementIdToCategoryIndex.Add(categoryIndex);
                elementIdToSubPath.Add(new List<string> {fileinfo.Name});
            }
        }
    }
}
