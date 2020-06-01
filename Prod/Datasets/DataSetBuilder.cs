using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using log4net;
using SharpNet.CPU;
using SharpNet.Pictures;
// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets
{
    // ReSharper disable once UnusedMember.Global
    public class DataSetBuilder
    {
        const string Header = "SHA1;OriginalPath;FileExtension;Width;Height;AverageColor;SuggestedId;Id;IdComment;SuggestedCancel;Cancel;CancelComment;InsertionDate;RemovedDate;ValidationDate";
        const string Header_IDM = "No;Action;NotUsed;SHA1;OriginalPath;FileExtension;SuggestedId;Id;SuggestedCancel;Cancel;ValidationDate;sha1Path;Date";
        #region private fields
        private static readonly ILog Log = LogManager.GetLogger(typeof(DataSetBuilder));
        private readonly IDictionary<string, DataSetBuilderEntry> _database = new Dictionary<string, DataSetBuilderEntry>();
        private readonly string _rootPath;
        private readonly CategoryHierarchy _root;
        #endregion
        #region constructor
        public DataSetBuilder(string rootPath, CategoryHierarchy root)
        {
            _rootPath = rootPath;
            _root = root;
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

        public void FindInvalidCancelCategories()
        {
            var root = CategoryHierarchy.ComputeRootNode();
            var invalids = new HashSet<string>();
            foreach (var d in _database.Values.Where(e=>!e.IsRemoved))
            {
                var cancel = d.Cancel;
                if (string.IsNullOrEmpty(cancel))
                {
                    continue;
                }
                if (root.ToPath(cancel) == null)
                {
                    invalids.Add(cancel);
                }
            }
            Log.Error(invalids.Count+"  invalid cancel: " + Environment.NewLine+string.Join(Environment.NewLine, invalids.OrderBy(s=>s)));
        }

        // ReSharper disable once UnusedMember.Global
        public void DetectDuplicate(double epsilon)
        {
            var entries = _database.Values.Where(e => !e.IsRemoved).OrderBy(e=>e.SuggestedId).ThenBy(e => e.AverageColor.DistanceToWhite).ToList();
            Log.Info("Looking for duplicates in database with "+entries.Count+" entries");
            var cache = new RGBColorFactoryWithCache(true);
            int duplicateFound = 0;
            int processed = 0;

            void ProcessElement(int i)
            {
                Interlocked.Increment(ref processed);
                if (processed % 10000 == 0)
                {
                    Log.Info(processed+ " entries processed(duplicate found: "+ duplicateFound+")");
                    FlushDatabase();
                }
                var iEntry = entries[i];
                if (iEntry.IsRemoved)
                {
                    return;
                }

                var iDistanceToWhite = iEntry.AverageColor.DistanceToWhite;
                var iBitmapContent = new Lazy<BitmapContent>(() => BitmapContent.ValueFomSingleRgbBitmap(iEntry.Path(_rootPath)));
                for (int j = i + 1; j < entries.Count; ++j)
                {
                    var jEntry = entries[j];
                    if (jEntry.IsRemoved)
                    {
                        continue;
                    }
                    if (iEntry.SuggestedId != jEntry.SuggestedId)
                    {
                        break;
                    }
                    var jDistanceToWhite = jEntry.AverageColor.DistanceToWhite;
                    if (Math.Abs(jDistanceToWhite - iDistanceToWhite) > 0.01)
                    {
                        break;
                    }
                    if (iEntry.IsDuplicate(iBitmapContent, jEntry, _rootPath, cache, epsilon))
                    {
                        int indexToRemove = iEntry.Count >= jEntry.Count ? i : j;
                        int indexToKeep = i+j- indexToRemove;
                        entries[indexToRemove].RemovedDate = DateTime.Now;
                        entries[indexToRemove].IdComment = "Duplicate of "+ entries[indexToKeep].SHA1;
                        entries[indexToKeep].ImportRelevantInfoFrom(entries[indexToRemove]);
                        Interlocked.Increment(ref duplicateFound);
                    }
                }
            }

            Parallel.For(0, entries.Count, ProcessElement);
            FlushDatabase();
            Log.Info(entries.Count+" entries processed (duplicate found: " + duplicateFound + ")");
        }

        public AbstractDataSet ExtractDataSet(Func<DataSetBuilderEntry, bool> accept)
        {
            var categoryNameToCount = new Dictionary<string,int>();
            var elementIdToPaths = new List<List<string>>();
            var elementIdToDescription = new List<string>();
            var elementIdToCategoryIndex = new List<int>();
            var entries = _database.Values.Where(e => !e.IsRemoved && accept(e) && (_root.ToPath(e.Cancel)?.Length??0)>=1

                                                      //?D
                                                      && _root.CategoryPathToCategoryName(_root.ToPath(e.Cancel)).Contains("used/star/")

                                                      ).OrderBy(e => e.SHA1).ToArray();

            var yExpected = new CpuTensor<float>(new [] {entries.Length, _root.RootPrediction().Length});
            for (var elementId = 0; elementId < entries.Length; elementId++)
            {
                var entry = entries[elementId];
                var categoryPath = _root.ToPath(entry.Cancel);
                Debug.Assert(categoryPath != null && categoryPath.Length >= 1);
                var categoryName = _root.CategoryPathToCategoryName(categoryPath);
                if (categoryNameToCount.ContainsKey(categoryName))
                {
                    ++categoryNameToCount[categoryName];
                }
                else
                {
                    categoryNameToCount[categoryName] = 1;
                }

                elementIdToPaths.Add(new List<string> {entry.Path(_rootPath)});
                elementIdToDescription.Add(entry.SHA1 + "_" + entry.Cancel);
                elementIdToCategoryIndex.Add(-1);
                var elementPrediction = _root.ExpectedPrediction(categoryPath);
                Debug.Assert(elementPrediction != null);
                Debug.Assert(elementPrediction.Length == yExpected.Shape[1]);
                for (int col = 0; col < elementPrediction.Length; ++col)
                {
                    yExpected.Set(elementId, col, elementPrediction[col]);
                }
            }
            Log.Info("found "+ elementIdToDescription.Count+" elements");
            Log.Info(string.Join(Environment.NewLine, categoryNameToCount.OrderBy(e=>e.Key).Select(e=>e.Key +" : "+e.Value)));
            var categoryDescription = Enumerable.Range(0, yExpected.Shape[1]).Select(i=>i.ToString()).ToArray();
            return new DirectoryDataSet(elementIdToPaths, elementIdToDescription, elementIdToCategoryIndex, yExpected, nameof(DataSetBuilder), 3, categoryDescription,
                new List<Tuple<float, float>> { Tuple.Create(147.02734f, 60.003986f), Tuple.Create(141.81636f, 51.15815f), Tuple.Create(130.15608f, 48.55502f) },
                ResizeStrategyEnum.ResizeToTargetSize);
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
            var cache = new RGBColorFactoryWithCache(true);
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

                    var bmp = BitmapContent.ValueFomSingleRgbBitmap(originalPath);
                    var entry = new DataSetBuilderEntry
                    {
                        SHA1 = sha1,
                        OriginalPath = originalPath,
                        Width = bmp.GetWidth(),
                        Height = bmp.GetHeight(),
                        AverageColor = bmp.AverageColor(cache),
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
        // ReSharper disable once UnusedMember.Global
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

        public static string DateTimeToString(DateTime? date)
        {
            return date.HasValue ? DateTimeToString(date.Value) : "";
        }
        public static string DateTimeToString(DateTime date)
        {
            return date.ToString(CultureInfo.InvariantCulture);
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
        private string AvgColorPath => Path.Combine(_rootPath, "DatasetAvgColor.csv");
        private void LoadDatabase()
        {
            Log.Info("Loading database "+_rootPath);
            _database.Clear();
            var lines = File.ReadAllLines(CsvPath).Skip(1).ToArray();
            var entries = new DataSetBuilderEntry[lines.Length];
            void LoadEntry(int entryIdx)
            {
                var splitted = lines[entryIdx].Split(';');
                Debug.Assert(splitted.Length == 15);
                int columnIdx = 0;
                var entry = new DataSetBuilderEntry
                {
                    SHA1 = splitted[columnIdx++],
                    OriginalPath = splitted[columnIdx++],
                    FileExtension = splitted[columnIdx++],
                    Width = int.Parse(splitted[columnIdx++]),
                    Height = int.Parse(splitted[columnIdx++]),
                    AverageColor= DataSetBuilderEntry.StringToAverageColor(splitted[columnIdx++]),
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
        // ReSharper disable once MemberCanBePrivate.Global
        public void FlushDatabase()
        {
            CreateBackupForDatabase();
            var sb = new StringBuilder();
            sb.Append(Header + Environment.NewLine);
            //foreach (var entry in _database.Values.OrderBy(e => e.Id).ThenBy(e => e.SuggestedId).ThenBy(e => e.InsertionDate))
            foreach (var entry in _database.Values.OrderBy(e => e.SHA1))
            {
                sb.Append(entry.AsCsv).Append(Environment.NewLine);
            }
            File.WriteAllText(CsvPath, sb.ToString());
        }
        private string Summary()
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

            var result = Count + " elements";
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
    }
}