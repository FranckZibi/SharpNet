using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public static class DogsVsCats
    {
        public static DirectoryDataSetLoader ValueOf(string directory, int height, int width, Logger logger)
        {
            return new DirectoryDataSetLoader(
                "",
                directory,
                logger,
                3, //Channels
                height,
                width,
                new[] {"cat", "dog"},
                false, //we do not ignore zero pixels
                ComputeCategoryIdDescriptionFullName);
        }

        private static void ComputeCategoryIdDescriptionFullName(
            string csvFileName,
            string dataDirectory,
            List<int> elementIdToCategoryId,
            List<string> elementIdToDescription,
            List<List<string>> elementIdToSubPath,
            Logger logger)
        {
            elementIdToCategoryId.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();

            foreach (var fileinfo in new DirectoryInfo(dataDirectory).GetFiles().Where(x=>PictureTools.IsPicture(x.FullName)))
            {
                var fullName = fileinfo.FullName;
                var description = Path.GetFileNameWithoutExtension(fullName);
                elementIdToDescription.Add(description);
                int categoryId = description.StartsWith("dog") ? 1 : 0;
                elementIdToCategoryId.Add(categoryId);
                elementIdToSubPath.Add(new List<string> {fileinfo.Name});
            }
        }
    }
}
