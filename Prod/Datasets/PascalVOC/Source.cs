using System.Xml;
using JetBrains.Annotations;

namespace SharpNet.Datasets.PascalVOC
{
    public class Source
    {
        #region public properties
        [NotNull] public string Database { get;}
        [NotNull] public string Annotation { get;}
        [NotNull] public string Image { get;}
        [NotNull] public string FlickrId { get;}
        #endregion

        #region constructor
        public Source([NotNull] string database, [NotNull]string annotation, [NotNull]string image, [NotNull]string flickrId)
        {
            Database = database;
            Annotation = annotation;
            Image = image;
            FlickrId = flickrId;
        }
        #endregion

        public static Source ValueOf(XmlNode node)
        {
            return new Source(Utils.GetString(node, "database"), Utils.GetString(node, "annotation"),Utils.GetString(node, "image"), Utils.GetString(node, "flickrid"));
        }
    }
}