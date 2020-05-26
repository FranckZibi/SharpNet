using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;
using SharpNet.Pictures;
// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets
{
    // ReSharper disable once UnusedMember.Global
    public class DataSetBuilder
    {

        #region class DataSetBuilderEntry
        public class DataSetBuilderEntry
        {
            public string SHA1 { get; set; }
            public string OriginalPath { get; set; }
            public int Width{ get; set; }
            public int Height { get; set; }
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
            //public Entry Clone()
            //{
            //    return new Entry
            //    {
            //        SHA1 = this.SHA1,
            //        OriginalPath = this.OriginalPath,
            //        FileExtension =  this.FileExtension,
            //        SuggestedId =  this.SuggestedId,
            //        Id = this.Id,
            //        IdComment = this.IdComment,
            //        SuggestedCancel =  this.SuggestedCancel,
            //        Cancel = this.Cancel,
            //        CancelComment  = this.CancelComment,
            //        InsertionDate =  this.InsertionDate,
            //        RemovedDate =  this.RemovedDate,
            //        ValidationDate = this.ValidationDate
            //    };
            //}
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
                       (ratio > expectedRatio/ (1 + toleranceInPercentage) );
            }

            public string GetCancelCayegory()
            {
                if (string.IsNullOrEmpty(Cancel))
                {
                    return "";
                }

                if (Cancel.StartsWith("gc"))
                {
                    return "gc";
                }
                if (Cancel.StartsWith("etoile"))
                {
                    return "etoile";
                }
                if (Cancel.StartsWith("etoile"))
                {
                    return "etoile";
                }
                if (Cancel.StartsWith("cad"))
                {
                    return "cad";
                }
                if (Cancel.StartsWith("mint"))
                {
                    return "mint";
                }
                if (Cancel.StartsWith("imprime"))
                {
                    return "imprime";
                }
                if (Cancel.StartsWith("pc"))
                {
                    return "pc";
                }

                return "unknown";
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
                    sb.Append(SuggestedId).Append(";");
                    sb.Append(Id).Append(";");
                    sb.Append(IdComment).Append(";");
                    sb.Append(SuggestedCancel).Append(";");
                    sb.Append(Cancel).Append(";");
                    sb.Append(CancelComment).Append(";");
                    sb.Append(DateTimeToString(InsertionDate)).Append(";");
                    sb.Append(DateTimeToString(RemovedDate)).Append(";");
                    sb.Append(DateTimeToString(ValidationDate));
                    return sb.ToString();
                }
            }

            public string AsCsv_IDM(string rootPath, int number)
            {
                var sb = new StringBuilder();
                sb.Append(number).Append(";");
                sb.Append("").Append(";");
                sb.Append("").Append(";");
                sb.Append(SHA1).Append(";");
                sb.Append(OriginalPath).Append(";");
                sb.Append(FileExtension).Append(";");
                sb.Append(SuggestedId).Append(";");
                sb.Append(Id).Append(";");
                sb.Append(SuggestedCancel).Append(";");
                sb.Append(Cancel).Append(";");
                sb.Append(DateTimeToString(ValidationDate)).Append(";");
                sb.Append(Path(rootPath)).Append(";");
                sb.Append(DateTimeToString(InsertionDate));
                return sb.ToString();
            }
        }
        #endregion


        public AbstractDataSet ExtractDataSet(Func<DataSetBuilderEntry, bool> accept, string[] categories, int maxCountByCategory)
        {
            var currentCount = new int[categories.Length];
            var elementIdToPaths = new List<List<string>>();
            var elementIdToDescription = new List<string>();
            var elementIdToCategoryIndex = new List<int>();

            foreach (var entry in _database.Values.Where(e => !e.RemovedDate.HasValue && accept(e)).OrderBy(e => e.SHA1))
            {
                var category = entry.GetCancelCayegory();
                var categoryIndex = Array.IndexOf(categories, category);
                if (categoryIndex < 0 || currentCount[categoryIndex] >= maxCountByCategory)
                {
                    continue;
                }
                ++currentCount[categoryIndex];
                elementIdToPaths.Add(new List<string> {entry.Path(_rootPath)});
                elementIdToDescription.Add(entry.SHA1);
                elementIdToCategoryIndex.Add(categoryIndex);
            }
            return new DirectoryDataSet(elementIdToPaths, elementIdToDescription, elementIdToCategoryIndex, nameof(DataSetBuilder), 3 , categories,
                new List<Tuple<float, float>> { Tuple.Create(147.02734f, 60.003986f), Tuple.Create(141.81636f, 51.15815f), Tuple.Create(130.15608f, 48.55502f) },
               ResizeStrategyEnum.ResizeToTargetSize);
        }

        const string Header = "SHA1;OriginalPath;FileExtension;Width;HeightSuggestedId;Id;IdComment;SuggestedCancel;Cancel;CancelComment;InsertionDate;RemovedDate;ValidationDate";
        #region private fields
        private static readonly ILog Log = LogManager.GetLogger(typeof(DataSetBuilder));
        private readonly IDictionary<string, DataSetBuilderEntry> _database = new Dictionary<string, DataSetBuilderEntry>();
        private readonly string _rootPath;
        #endregion
        #region constructor
        public DataSetBuilder(string rootPath)
        {
            _rootPath = rootPath;
            if (!Directory.Exists(_rootPath))
            {
                Directory.CreateDirectory(_rootPath);
            }
            if (!File.Exists(CsvPath))
            {
                File.WriteAllText(CsvPath, Header + Environment.NewLine);
            }
            LoadDatabase();
        }
        #endregion

        public string Summary()
        {
            int removed = 0;
            int withCancel = 0;
            int withSuggestedCancel = 0;
            int withSuggestedId = 0;
            int withId = 0;
            foreach (var e in _database)
            {
                if (e.Value.RemovedDate.HasValue)
                {
                    ++removed;
                    continue;
                }

                if (!string.IsNullOrEmpty(e.Value.Id))
                {
                    ++withId;
                }
                else
                {
                    if (!string.IsNullOrEmpty(e.Value.SuggestedId))
                    {
                        ++withSuggestedId;
                    }

                }
                if (!string.IsNullOrEmpty(e.Value.Cancel))
                {
                    ++withCancel;
                }
                else
                {
                    if (!string.IsNullOrEmpty(e.Value.SuggestedCancel))
                    {
                        ++withSuggestedCancel;
                    }
                }
            }

            var result = Count+ " elements";
            if (removed != 0)
            {
                result += " ;removed=" + removed;
            }
            if (withId != 0)
            {
                result += " ;with Id=" + withId;
            }
            if (withSuggestedId != 0)
            {
                result += " ;with suggested Id=" + withSuggestedId;
            }
            if (withCancel != 0)
            {
                result += " ;with cancel=" + withCancel;
            }
            if (withSuggestedCancel != 0)
            {
                result += " ;with suggested cancel=" + withSuggestedCancel;
            }
            return result;
        }

        /// <summary>
        /// load all images in 'path' into the database
        /// return the number of images added in the database
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        // ReSharper disable once UnusedMember.Global
        public void AddAllFilesInPath(string path)
        {
            Log.Info("Adding files from " + path+" ...");
            var initialCount = Count;

            var filesInPath = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories);
            Log.Info("There are "+filesInPath.Length+" files in " + path);
            void ProcessFile(string originalPath)
            {
                try
                {
                    if (!PictureTools.IsPicture(originalPath))
                    {
                        return;
                    }
                    var sha1 = Utils.ImagePathToSHA1(originalPath);
                    var fileExtension = Path.GetExtension(originalPath).TrimStart('.');
                    var entry = new DataSetBuilderEntry
                    {
                        SHA1 = sha1,
                        OriginalPath = originalPath,
                        FileExtension = fileExtension,
                        SuggestedId = ExtractSuggestedId(originalPath),
                        InsertionDate = DateTime.Now
                    };
                    lock (_database)
                    {
                        if (_database.ContainsKey(sha1))
                        {
                            return;
                        }
                        _database[sha1] = entry;
                    }
                    var destFileName = entry.Path(_rootPath);
                    var destDirectory = Path.GetDirectoryName(destFileName);
                    if (!Directory.Exists(destDirectory))
                    {
                        Directory.CreateDirectory(destDirectory);
                    }

                    File.Copy(originalPath, destFileName);
                }
                catch (Exception e)
                {
                    Log.Error("fail to process image "+ originalPath, e);
                }

            }

            Parallel.For(0, filesInPath.Length, i=> ProcessFile(filesInPath[i]));

            if (Count > initialCount)
            {
                FlushDatabase();
            }
            Log.Info("Added "+(Count-initialCount)+" files from "+path);
        }
        public void FlushDatabase()
        {
            CreateBackupForDatabase();
            var sb = new StringBuilder();
            sb.Append(Header + Environment.NewLine);
            foreach (var entry in _database.Values.OrderBy(e => e.Id).ThenBy(e => e.SuggestedId).ThenBy(e => e.InsertionDate))
            {
                sb.Append(entry.AsCsv).Append(Environment.NewLine);
            }
            File.WriteAllText(CsvPath, sb.ToString());
        }


        private const string Header_IDM = "No;Action;NotUsed;SHA1;OriginalPath;FileExtension;SuggestedId;Id;SuggestedCancel;Cancel;ValidationDate;sha1Path;Date";

        public void CreateIDM(string idmPath, Func<DataSetBuilderEntry, bool> accept)
        {
            var sb = new StringBuilder();
            sb.Append(Header_IDM).Append(Environment.NewLine);
            int number = 0;
            foreach (var e in _database)
            {
                if (!accept(e.Value))
                {
                    continue;
                }

                ++number;
                sb.Append(e.Value.AsCsv_IDM(_rootPath, number)).Append(Environment.NewLine);
            }
            File.WriteAllText(idmPath, sb.ToString());
        }


        ///// <summary>
        ///// load all images in 'path' into the database
        ///// return the number of images added in the database
        ///// </summary>
        ///// <returns></returns>
        //public void CopyToDatabaseWithoutDuplicate(string newDatabasePath)
        //{
        //    var newDatabase = new DataSetBuilder(newDatabasePath);
        //    var oldDatabaseContent = _database.ToList();
        //    Log.Info("Copy database to " + newDatabasePath + " ...");
        //    Log.Info("There are " + oldDatabaseContent.Count+ " element in " + _rootPath);

        //    int nbProcessed = 0;

        //    void ProcessFile(int idx)
        //    {
        //        try
        //        {
        //            Interlocked.Increment(ref nbProcessed);
        //            if (nbProcessed % 1000 == 0)
        //            {
        //                Log.Info( Math.Round( (100.0* nbProcessed) /oldDatabaseContent.Count, 2)+"%");
        //            }

        //            var originalItem = oldDatabaseContent[idx];
        //            var oldEntry = originalItem.Value;
        //            var oldPath = oldEntry.Path(_rootPath);
        //            var newEntry = oldEntry.Clone();
        //            var newSHA1 = ImagePathToSHA1(oldPath);
        //            newEntry.SHA1 = newSHA1;
        //            var newPath = newEntry.Path(newDatabasePath);

        //            lock (_database)
        //            {
        //                if (newDatabase._database.ContainsKey(newSHA1))
        //                {
        //                    if (!string.IsNullOrEmpty(newEntry.Cancel))
        //                    {
        //                        newDatabase._database[newSHA1].Cancel = newEntry.Cancel;
        //                    }
        //                    return;
        //                }
        //                else
        //                {
        //                    newDatabase._database[newSHA1] = newEntry;
        //                }
        //            }
        //            var destDirectory = Path.GetDirectoryName(newPath);
        //            if (!Directory.Exists(destDirectory))
        //            {
        //                Directory.CreateDirectory(destDirectory);
        //            }

        //            File.Copy(oldPath, newPath);
        //        }
        //        catch (Exception e)
        //        {
        //            Log.Error("fail to process image " + oldDatabaseContent[idx].Value.Path(_rootPath), e);
        //        }

        //    }
        //    Parallel.For(0, oldDatabaseContent.Count, ProcessFile);
        //    newDatabase.FlushDatabase();
        //    Log.Info("new database contains " + newDatabase.Count +" elements");
        //    Log.Info("original database contains " + Count +" elements");
        //}

        private static string DateTimeToString(DateTime? date)
        {
            return date.HasValue ? date.Value.ToString(CultureInfo.InvariantCulture) : "";
        }
        private static DateTime? StringToDateTime(string str)
        {
            if (string.IsNullOrWhiteSpace(str))
            {
                return null;
            }
            return DateTime.Parse(str, CultureInfo.InvariantCulture);
        }
        private static string ExtractSuggestedId(string originalPath)
        {
            var f = new FileInfo(originalPath);
            if (f.Directory != null)
            {
                return f.Directory.Name;
            }
            return "";
        }
        private void CreateBackupForDatabase()
        {
            File.Copy(CsvPath, CsvPath + "_backup_" + DateTime.Now.Ticks);
        }
        private int Count => _database.Count;
        private string CsvPath => Path.Combine(_rootPath, "Dataset.csv");
        private void LoadDatabase()
        {
            Log.Info("Loading database "+_rootPath);
            _database.Clear();
            var lines = File.ReadAllLines(CsvPath).Skip(1).ToArray();
            var entries = new DataSetBuilderEntry[lines.Length];
            void LoadEntry(int entryIdx)
            {
                var splitted = lines[entryIdx].Split(';');
                Debug.Assert(splitted.Length == 14);
                int columnIdx = 0;
                var entry = new DataSetBuilderEntry
                {
                    SHA1 = splitted[columnIdx++],
                    OriginalPath = splitted[columnIdx++],
                    FileExtension = splitted[columnIdx++],
                    Width = int.Parse(splitted[columnIdx++]),
                    Height = int.Parse(splitted[columnIdx++]),
                    SuggestedId = splitted[columnIdx++],
                    Id = splitted[columnIdx++],
                    IdComment = splitted[columnIdx++],
                    SuggestedCancel = splitted[columnIdx++],
                    Cancel = splitted[columnIdx++],
                    CancelComment = splitted[columnIdx++],
                    // ReSharper disable once PossibleInvalidOperationException
                    InsertionDate = StringToDateTime(splitted[columnIdx++]).Value,
                    RemovedDate = StringToDateTime(splitted[columnIdx++]),
                    ValidationDate = StringToDateTime(splitted[columnIdx])
                };
                entries[entryIdx] = entry;
            }
            Parallel.For(0, lines.Length, LoadEntry);
            foreach (var e in entries)
            {
                _database[e.SHA1] = e;
            }
            Log.Info(Summary());
        }
    }
}