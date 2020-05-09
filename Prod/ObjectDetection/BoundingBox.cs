using System;
using System.Diagnostics;
using System.Linq;
using System.Xml;

namespace SharpNet.ObjectDetection
{
    public class BoundingBox
    {
        #region private fields
        private readonly double _height;
        private readonly double _width;
        private readonly double _colCenter;
        private readonly double _rowCenter;
        #endregion

        #region constructor
        public BoundingBox(double height, double width, double colCenter, double rowCenter)
        {
            Debug.Assert(new[]{height,width,colCenter,rowCenter}.Max()<=1.0);
            Debug.Assert(new[]{height,width,colCenter,rowCenter}.Min()>=0.0);
            _height = height;
            _width = width;
            _colCenter = colCenter;
            _rowCenter = rowCenter;
        }
        #endregion

        /// <summary>
        /// Intersection over union
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public double IoU(BoundingBox other)
        {
            var intersection = Intersection(other);
            if (intersection < 1e-6)
            {
                return 0.0;
            }
            return intersection / Union(other);
        }


        public static BoundingBox FromPascalVOC(XmlNode node, int imageHeight, int imageWidth)
        {
            Debug.Assert(imageHeight >= 1);
            Debug.Assert(imageWidth >= 1);
            int colStart = Utils.GetInt(node, "xmin", -1);
            int colEnd = Utils.GetInt(node, "xmax", -1);
            int rowStart = Utils.GetInt(node, "ymin", -1);
            int rowEnd = Utils.GetInt(node, "ymax", -1);
            Debug.Assert(new[] { colStart, colEnd, rowStart, rowEnd }.Min() >= 0);
            double boxWidth = (colEnd - colStart + 1.0) / imageWidth;
            double boxHeight = (rowEnd - rowStart + 1.0) / (2.0*imageHeight);
            double colCenter = (colEnd + colStart + 1.0) / (2.0*imageWidth);
            double rowCenter = (rowEnd + rowStart + 1.0) / (2.0*imageHeight);
            return new BoundingBox(boxHeight, boxWidth, colCenter, rowCenter);
        }


        public double Intersection(BoundingBox other)
        {
            double maxLeft = Math.Max(Left, other.Left);
            double minRight = Math.Min(Right, other.Right);
            if (maxLeft >= minRight)
            {
                return 0.0;
            }
            double maxTop = Math.Max(Top, other.Top);
            double minBottom = Math.Min(Bottom, other.Bottom);
            if (maxTop >= minBottom)
            {
                return 0.0;
            }
            return (minRight - maxLeft) * (minBottom - maxLeft);
        }

        public double Union(BoundingBox other)
        {
            return Area + other.Area - Intersection((other));
        }

        public double Area => _height * _width;

        public double Height => _height;
        public double Width => _width;
        public double ColCenter => _colCenter;
        public double RowCenter => _rowCenter;
        public double Right => ColCenter+Width/2;
        public double Left => ColCenter-Width/2;
        public double Top => RowCenter - Height / 2;
        public double Bottom => RowCenter + Height/ 2;



        public override string ToString()
        {
            return "Center: ("+Math.Round(RowCenter,3)+", "+ Math.Round(ColCenter, 3)+") Height:"+  Math.Round(Height, 3)+" Width: "+ Math.Round(Width, 3);
        }
        public bool Equals(BoundingBox other, double epsilon)
        {
            return    Math.Abs(_height - other._height) <= epsilon
                   && Math.Abs(_width - other._width) <= epsilon
                   && Math.Abs(_colCenter - other._colCenter) <= epsilon
                   && Math.Abs(_rowCenter - other._rowCenter) <= epsilon
                ;
        }
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }
            if (ReferenceEquals(this, obj))
            {
                return true;
            }
            if (obj.GetType() != GetType())
            {
                return false;
            }
            return Equals((BoundingBox)obj, 1e-6);
        }
        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }
    }
}
