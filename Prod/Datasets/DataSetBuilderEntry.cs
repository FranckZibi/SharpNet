using System;
using System.Diagnostics;
using System.Linq;
using System.Text;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public class DataSetBuilderEntry
    {
        public string SHA1 { get; set; }
        public string OriginalPath { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public RGBColor AverageColor { get; set; }
        public string FileExtension { get; set; }
        public string SuggestedId { get; set; }
        public string Id { get; set; }
        public string IdComment { get; set; }
        public string SuggestedCancel { get; set; }
        public string Cancel { get; set; }
        public string CancelComment { get; set; }
        public DateTime InsertionDate { get; set; }
        public DateTime? RemovedDate { get; set; }
        public DateTime? ValidationDate { get; set; }

        /// <summary>
        /// the path of the item in the HD
        /// </summary>
        /// <param name="rootPath"></param>
        /// <returns></returns>
        public string Path(string rootPath)
        {
            var subDirectory = SHA1.Substring(0, 2);
            var fileName = SHA1.Substring(2);
            if (!string.IsNullOrEmpty(FileExtension))
            {
                fileName += "." + FileExtension;
            }

            return System.IO.Path.Combine(rootPath, subDirectory, fileName);
        }

        public void ImportRelevantInfoFrom(DataSetBuilderEntry e)
        {

            if (string.IsNullOrEmpty(Id) && !string.IsNullOrEmpty(e.Id))
            {
                Id = e.Id;
            }
            if (string.IsNullOrEmpty(SuggestedId) && !string.IsNullOrEmpty(e.SuggestedId))
            {
                SuggestedId = e.SuggestedId;
            }
            if (string.IsNullOrEmpty(Cancel) && !string.IsNullOrEmpty(e.Cancel))
            {
                Cancel = e.Cancel;
            }
            if (string.IsNullOrEmpty(SuggestedCancel) && !string.IsNullOrEmpty(e.SuggestedCancel))
            {
                SuggestedCancel = e.SuggestedCancel;
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="expectedRatio"></param>
        /// <param name="toleranceInPercentage">0.10 mean 10% tolerance </param>
        /// <returns></returns>
        public bool HasExpectedWidthHeightRatio(double expectedRatio, double toleranceInPercentage)
        {
            if (Height <= 0)
            {
                return false;
            }

            var ratio = Width / (double) Height;
            return (ratio < (1 + toleranceInPercentage) * expectedRatio) &&
                   (ratio > expectedRatio / (1 + toleranceInPercentage));
        }

        public int Count => Width * Height;





        public bool IsDuplicate(Lazy<BitmapContent> thisBitmapContent, DataSetBuilderEntry b, string rootPath, RGBColorFactoryWithCache cache, double epsilon)
        {
            if (IsRemoved || b.IsRemoved)
            {
                return false;
            }
            if (!string.IsNullOrEmpty(SuggestedId) && !string.IsNullOrEmpty(b.SuggestedId) && SuggestedId != b.SuggestedId)
            {
                return false;
            }
            if (!string.IsNullOrEmpty(Cancel) && !string.IsNullOrEmpty(b.Cancel) && Cancel != b.Cancel)
            {
                return false;
            }
            if (!string.IsNullOrEmpty(SuggestedCancel) && !string.IsNullOrEmpty(b.SuggestedCancel) && SuggestedCancel != b.SuggestedCancel)
            {
                return false;
            }
            var thisRatio = ((double) Width) / Height;
            if (!HasExpectedWidthHeightRatio(thisRatio, 0.01))
            {
                return false;
            }
            if ((Width > b.Width && Height < b.Height) || (Width < b.Width && Height > b.Height))
            {
                return false;
            }
            if (AverageColor.ColorDistance(b.AverageColor) > 0.015)
            {
                return false;
            }

            return thisBitmapContent.Value.IsDuplicate(BitmapContent.ValueFomSingleRgbBitmap(b.Path(rootPath)), epsilon, cache);
        }

        public string AsCsv
        {
            get
            {
                var sb = new StringBuilder();
                sb.Append(SHA1).Append(";");
                sb.Append(OriginalPath).Append(";");
                sb.Append(FileExtension).Append(";");
                sb.Append(Width).Append(";");
                sb.Append(Height).Append(";");
                sb.Append(ColorToString(AverageColor)).Append(";");
                sb.Append(SuggestedId).Append(";");
                sb.Append(Id).Append(";");
                sb.Append(IdComment).Append(";");
                sb.Append(SuggestedCancel).Append(";");
                sb.Append(Cancel).Append(";");
                sb.Append(CancelComment).Append(";");
                sb.Append(DataSetBuilder.DateTimeToString(InsertionDate)).Append(";");
                sb.Append(DataSetBuilder.DateTimeToString(RemovedDate)).Append(";");
                sb.Append(DataSetBuilder.DateTimeToString(ValidationDate));
                return sb.ToString();
            }
        }

        public static RGBColor StringToAverageColor(string averageColorAsString)
        {
            if (string.IsNullOrEmpty(averageColorAsString))
            {
                return null;
            }

            var splitted = averageColorAsString.Split('|').Select(byte.Parse).ToArray();
            Debug.Assert(splitted.Length == 3);
            return new RGBColor(splitted[0], splitted[1], splitted[2]);
        }

        public string AsCsv_IDM(string rootPath, int number)
        {
            var sb = new StringBuilder();
            sb.Append(number).Append(";");
            
            //sb.Append("").Append(";");
            sb.Append(CancelComment).Append(";");

            sb.Append("").Append(";");
            sb.Append(SHA1).Append(";");
            sb.Append(OriginalPath).Append(";");
            sb.Append(FileExtension).Append(";");
            sb.Append(SuggestedId).Append(";");
            sb.Append(Id).Append(";");
            sb.Append(SuggestedCancel).Append(";");
            sb.Append(Cancel).Append(";");
            sb.Append(DataSetBuilder.DateTimeToString(ValidationDate)).Append(";");
            sb.Append(Path(rootPath)).Append(";");
            sb.Append(DataSetBuilder.DateTimeToString(InsertionDate));
            return sb.ToString();
        }

        public RGBColor ComputeAverageColor(string rootPath, RGBColorFactoryWithCache cache)
        {
            return IsRemoved ? null : BitmapContent.ValueFomSingleRgbBitmap(Path(rootPath)).AverageColor(cache);
        }

        private static string ColorToString(RGBColor c)
        {
            return c == null ? "" : c.Red + "|" + c.Green + "|" + c.Blue;
        }

        public bool IsRemoved => RemovedDate.HasValue;
    }
}