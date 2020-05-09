using System.Xml;
using JetBrains.Annotations;

namespace SharpNet.Datasets.PascalVOC
{
    public class Owner
    {
        #region public properties
        [NotNull] public string FlickrId { get; }
        [NotNull] public string Name { get; }
        #endregion

        #region constructor
        public Owner([NotNull] string flickrId, [NotNull] string name)
        {
            FlickrId = flickrId;
            Name = name;
        }
        #endregion

        public static Owner ValueOf(XmlNode node)
        {
            return new Owner(Utils.GetString(node, "flickrid"), Utils.GetString(node, "name"));
        }
    }
}