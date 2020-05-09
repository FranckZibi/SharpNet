using System;
using System.Xml;
using SharpNet.ObjectDetection;

namespace SharpNet.Datasets.PascalVOC
{
    public class ObjectDescription
    {
        #region public properties
        public int CategoryId { get;}
        public string Pose { get;}
        public bool Truncated { get;}
        public bool Difficult { get;}
        public BoundingBox Box { get;}
        #endregion

        #region constructor
        public ObjectDescription(int categoryId, string pose, bool truncated, bool difficult, BoundingBox boundingBox)
        {
            CategoryId = categoryId;
            Pose = pose;
            Truncated = truncated;
            Difficult = difficult;
            Box = boundingBox;
        }
        #endregion

        public static ObjectDescription ValueOf(XmlNode node, int imageHeight, int imageWidth)
        {
            var categoryName = Utils.GetString(node, "name");
            var categoryId = Array.IndexOf(PascalVOCDataSet.CategoryIndexToDescription, categoryName);
            if (categoryId < 0)
            {
                throw new ArgumentException("invalid category "+categoryName);
            }

            return new ObjectDescription(
                categoryId,
                Utils.GetString(node, "pose"),
                Utils.GetBool(node, "truncated", false),
                Utils.GetBool(node, "difficult", false),
                BoundingBox.FromPascalVOC(node.SelectSingleNode("bndbox"), imageHeight, imageWidth)
            );
        }
    }
}