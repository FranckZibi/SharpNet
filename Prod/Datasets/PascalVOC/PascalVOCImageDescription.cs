using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Xml;
using JetBrains.Annotations;

namespace SharpNet.Datasets.PascalVOC
{
    public class PascalVOCImageDescription
    {
        #region public properties
        [NotNull] public string Folder { get; }
        [NotNull] public string FileName { get; }
        [NotNull] public Source Source { get; }
        [NotNull] public Owner Owner { get; }
        /// <summary>
        /// Size[0] = height
        /// Size[1] = width
        /// Size[2] = channels
        /// </summary>
        public int[] Size { get; }
        public bool Segmented { get; }
        public List<ObjectDescription> Objects { get; }
        #endregion

        #region constructor
        public PascalVOCImageDescription([NotNull]string folder, [NotNull]string fileName, [NotNull]Source source, [NotNull]Owner owner, int[] size, bool segmented, List<ObjectDescription> objects)
        {
            Folder = folder;
            FileName = fileName;
            Source = source;
            Owner = owner;
            Size = size;
            Segmented = segmented;
            Objects = objects;
        }
        #endregion

        public static PascalVOCImageDescription ValueOf(string filePath)
        {
            var doc = new XmlDocument();
            doc.Load(filePath);
            var node = doc.SelectSingleNode("annotation");

            //we retrieve the shape of the picture (height, width, channelCount)
            var sizeNode = node.SelectSingleNode("size");
            var imageHeight = Utils.GetInt(sizeNode, "height", -1);
            var imageWidth = Utils.GetInt(sizeNode, "width", -1);
            var size = new [] { imageHeight, imageWidth, Utils.GetInt(sizeNode, "depth", -1)};
            Debug.Assert(size.Min() >= 1);

            var objects = new List<ObjectDescription>();
            foreach (XmlNode objectNode in node.SelectNodes("object"))
            {
                objects.Add(ObjectDescription.ValueOf(objectNode, imageHeight, imageWidth));
            }

            return new PascalVOCImageDescription(
                Utils.GetString(node, "folder"), 
                Utils.GetString(node, "filename"), 
                Source.ValueOf(node.SelectSingleNode("source")),
                Owner.ValueOf(node.SelectSingleNode("owner")),
                size,
                Utils.GetBool(node, "segmented", false),
                objects
            );
        }

        public int Height => Size[0];
        public int Width => Size[1];
        public int Channels => Size[2];

    }
}