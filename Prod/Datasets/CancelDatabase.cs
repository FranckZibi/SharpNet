using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using log4net;
using SharpNet.CPU;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNet.Pictures;
// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets
{
    // ReSharper disable once UnusedMember.Global
    public class CancelDatabase
    {
        const string Header = "SHA1;OriginalPath;FileExtension;Width;Height;AverageColor;SuggestedId;Id;IdComment;SuggestedCancel;Cancel;CancelComment;InsertionDate;RemovedDate;ValidationDate";
        const string Header_IDM = "No;Action;NotUsed;SHA1;OriginalPath;FileExtension;SuggestedId;Id;SuggestedCancel;Cancel;ValidationDate;sha1Path;Date";
        #region private fields
        private static readonly ILog Log = LogManager.GetLogger(typeof(CancelDatabase));
        private readonly IDictionary<string, CancelDatabaseEntry> _database = new Dictionary<string, CancelDatabaseEntry>();
        private readonly string _rootPath;
        #endregion
        public static readonly CategoryHierarchy Hierarchy = ComputeRootNode();
        #region constructor
        public CancelDatabase()
        {
            _rootPath = Path.Combine(NetworkConfig.DefaultDataDirectory, "Cancel");
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
            var invalids = new HashSet<string>();
            foreach (var d in _database.Values.Where(e=>!e.IsRemoved))
            {
                var cancel = d.Cancel;
                if (string.IsNullOrEmpty(cancel))
                {
                    continue;
                }
                if (ToPath(cancel) == null)
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

        /// <summary>
        /// TODO : add tests
        /// </summary>
        /// <param name="expected"></param>
        /// <param name="predicted"></param>
        /// <returns></returns>
        // ReSharper disable once MemberCanBePrivate.Global
        public static bool IsOkPrediction(string expected, string predicted)
        {
            expected = expected.Replace("_bleue", "").Replace("_bleu", "").Replace("_rouge", "");
            if (string.IsNullOrEmpty(expected))
            {
                return true;
            }
            if (string.IsNullOrEmpty(predicted))
            {
                return false;
            }
            if (expected.Equals(predicted))
            {
                return true;
            }
            if (expected.StartsWith("gc") && expected.Length > 6)
            {
                expected = expected.Substring(0, 6);
            }
            predicted = predicted.Replace("cad_imprime","imprime_rouge").Replace("anchor","ancre").Replace("cad_passe", "passe").Replace("cad_barcelona", "barcelona");
            if (expected.StartsWith("used"))
            {
                return !predicted.Equals("mint");
            }
            if (expected.StartsWith("amb"))
            {
                return predicted.StartsWith("amb");
            }
            if (expected.StartsWith("passe"))
            {
                return predicted.StartsWith("passe");
            }
            if (expected.Length > predicted.Length)
            {
                return false;
            }

            for (int i = 0; i < expected.Length; ++i)
            {
                if (predicted[i] != expected[i] && expected[i] != '*')
                {
                    return false;
                }
            }
            return true;
        }

        // ReSharper disable once UnusedMember.Global
        public void UpdateSuggestedCancelForAllDatabase(string modelName)
        {
            //foreach (var e in _database.Values.Where(e =>!e.IsRemoved))
            //{
            //    e.CancelComment = "";
            //    if (IsValidNonEmptyCancel(e.Cancel) && IsValidNonEmptyCancel(e.SuggestedCancel))
            //    {
            //        e.CancelComment = IsOkPrediction(e.Cancel, e.SuggestedCancel)?"OK":"KO";
            //    }
            //}
            //FlushDatabase();return;

            using var network =Network.LoadTrainedNetworkModel(Path.Combine(NetworkConfig.DefaultWorkingDirectory, "Cancel"), modelName);

            //foreach (var e in _database.Values.Where(e => !e.IsRemoved))
            //{
            //    e.SuggestedCancel = "";
            //}

            using var dataSet = ExtractDataSet(e => IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            var p = network.Predict(dataSet, Math.Min(32, dataSet.Count));
            for (int elementId = 0; elementId < p.Shape[0]; ++elementId)
            {
                var rowWithPrediction = p.RowSlice(elementId, 1);
                var sha1 = dataSet.ElementIdToDescription(elementId);
                var predictionWithProba = Hierarchy.ExtractPredictionWithProba(rowWithPrediction.AsReadonlyFloatCpuContent);
                var predictedCancelName = predictionWithProba.Item1;
                var predictedCancelProba = predictionWithProba.Item2;
                var e = _database[sha1];
                e.SuggestedCancel = predictedCancelName;
                e.CancelComment = "";
                e.IdComment = predictedCancelProba.ToString(CultureInfo.InvariantCulture);
                if (IsValidNonEmptyCancel(e.Cancel) && IsValidNonEmptyCancel(e.SuggestedCancel))
                {
                    e.CancelComment = IsOkPrediction(e.Cancel, e.SuggestedCancel) ? "OK" : "KO";
                }
            }
            FlushDatabase();
        }


        public static Network GetDefaultNetwork()
        {
            var network = Network.LoadTrainedNetworkModel(Path.Combine(NetworkConfig.DefaultWorkingDirectory, "Cancel"), "efficientnet-b0_Cancel_400_470_20200715_2244_630");
            return network;
        }

        public static List<Tuple<string, double>> PredictCancelsWithProba(Network network, List<string> picturePaths)
        {
            var result = new List<Tuple<string, double>>();
            using var dataSet = DirectoryDataSet.FromFiles(picturePaths, Hierarchy.RootPrediction().Length, CancelMeanAndVolatilityForEachChannel, ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            var p = network.Predict(dataSet, Math.Min(32, dataSet.Count));
            for (int elementId = 0; elementId < p.Shape[0]; ++elementId)
            {
                var rowWithPrediction = p.RowSlice(elementId, 1).AsReadonlyFloatCpuContent;
                result.Add(Hierarchy.ExtractPredictionWithProba(rowWithPrediction));
            }
            return result;
        }

        public static bool IsValidNonEmptyCancel(string cancelName)
        {

            if (string.IsNullOrEmpty(cancelName))
            {
                return false;
            }
            var path = ToPath(cancelName);
            return path != null && path.Length >= 1;
        }

        [SuppressMessage("ReSharper", "EnforceIfStatementBraces")]
        public static string[] ToPath(string cancelName)
        {
            cancelName = cancelName.Replace("_bleue", "").Replace("_bleu", "").Replace("_rouge", "");
            if (string.IsNullOrEmpty(cancelName)) return new string[0];
            if (cancelName == "mint") return new[] { "mint" };
            if (cancelName.StartsWith("used")) return new[] { "used" };
            if (cancelName.StartsWith("cad_perle")) return new[] { "used", "cad_perle" };
            if (cancelName.StartsWith("cad_octo")) return new[] { "used", "cad_octo" };
            if (cancelName.StartsWith("cad_ondule")) return new[] { "used", "cad_ondule" };
            if (cancelName.StartsWith("passe")) return new[] { "used", "cad_passe" };
            if (cancelName == "imprime") return new[] { "used", "cad_imprime" };
            if (cancelName.StartsWith("barcelona")) return new[] { "used", "cad_barcelona" };
            if (cancelName == "cad") return new[] { "used", "cad" };
            if (cancelName == "preo1893") return new[] { "used", "preo1893" };
            if (cancelName == "typo") return new[] { "used", "typo" };
            if (cancelName == "grille_ss_fin") return new[] { "used", "grille_ss_fin" };
            if (cancelName == "grille") return new[] { "used", "grille" };
            if (cancelName == "gros_points") return new[] { "used", "gros_points" };
            if (cancelName == "asna") return new[] { "used", "asna" };
            if (cancelName.StartsWith("amb")) return new[] { "used", "amb" };
            if (cancelName.StartsWith("ancre")) return new[] { "used", "anchor" };
            if (cancelName.StartsWith("plume")) return new[] { "used", "plume" };

            if (cancelName.StartsWith("etoile"))
            {
                if (cancelName == "etoile") return new[] { "used", "star" };
                if (cancelName.StartsWith("etoile_pleine")) return new[] { "used", "star", "full" };
                if (cancelName.StartsWith("etoile_evidee")) return new[] { "used", "star", "empty" };
                if (cancelName.Length == 7)
                {
                    if (cancelName.Equals("etoile*")) return new[] { "used", "star", "1digit" };
                    return new[] { "used", "star", "1digit", cancelName[6].ToString() };
                }
                if (cancelName.Length == 8)
                {
                    if (cancelName.Equals("etoile**")) return new[] { "used", "star", "2digits" };
                    return new[] { "used", "star", "2digits", cancelName[6].ToString(), cancelName[7].ToString() };
                }
                return null;
            }

            if (cancelName.StartsWith("gc") || cancelName.StartsWith("pc"))
            {
                var result = new List<string> { "used", cancelName.Substring(0, 2) };
                if (cancelName.Length == 2)
                {
                    return result.ToArray();
                }

                var subCategory = "1digit";
                if (cancelName.Length >= 4)
                {
                    subCategory = (Math.Min(cancelName.Length, 6) - 2) + "digits";
                }
                result.Add(subCategory);
                if (cancelName.Skip(2).All(c => c == '*'))
                {
                    return result.ToArray();
                }

                for (int i = 2; i < Math.Min(cancelName.Length, 6); ++i)
                {
                    if (!char.IsDigit(cancelName[i]) && cancelName[i] != '*') return null;
                    result.Add(cancelName[i].ToString());
                }
                return result.ToArray();
            }
            return null;
        }

        // ReSharper disable once UnusedMember.Global
        public void CreatePredictionFile(string outputFile)
        {
            var sb = new StringBuilder();
            sb.Append("ElementId;ElementDescription;Expected;Predicted;Correct?;5;6;7;8;9;10;Path;Date;11;12").Append(Environment.NewLine);
            int count = 0;
            var totalCount = new Dictionary<string, int>();
            var countKO = new Dictionary<string, int>();
            foreach (var e in _database.Values.Where(
                    e=>!e.IsRemoved 
                       && IsValidNonEmptyCancel(e.Cancel)
                       && IsValidNonEmptyCancel(e.SuggestedCancel)
                    //&& e.SHA1 == "38485D1A3622A6E5612FBB9BD505A039C746B291"
                        ))
            {
                var isOkPrediction = IsOkPrediction(e.Cancel, e.SuggestedCancel);
                var elements = new List<string>
                {
                    count.ToString(), 
                    e.SHA1,
                    e.Cancel,
                    e.SuggestedCancel,
                    isOkPrediction? "OK" : "KO",
                    e.IdComment,
                    "",
                    "",
                    "",
                    "",
                    "",
                    e.Path(_rootPath),
                    "",
                    ""
                };
                sb.Append(string.Join(";", elements)).Append(Environment.NewLine);

                ++count;

                var prefix = e.Cancel.Substring(0, 2);
                if (!totalCount.ContainsKey(prefix))
                {
                    totalCount[prefix] = 0;
                    countKO[prefix] = 0;

                }
                ++totalCount[prefix];
                if (!isOkPrediction)
                {
                    ++countKO[prefix];
                }
            }

            //We compute stats about the observed errors
            var errorStats = new List<Tuple<string, double>>();
            foreach (var prefix in totalCount.Keys.ToList())
            {
                int total = totalCount[prefix];
                int errors = countKO[prefix];
                errorStats.Add(Tuple.Create(prefix, ((double)errors)/ total));
            }
            var totalSum = totalCount.Values.Sum();
            var totalErrors = countKO.Values.Sum();
            IModel.Log.Info("Errors (total):" + Math.Round(100 * ((double)totalErrors) / totalSum, 2) + "% (" + totalErrors + "/" + totalSum + ")");
            IModel.Log.Info(string.Join(Environment.NewLine, errorStats.OrderByDescending(e=>e.Item2).ThenByDescending(e=>totalCount[e.Item1]).Select(e=>e.Item1+":"+Math.Round(100*e.Item2,2)+"% ("+ countKO[e.Item1]+"/"+ totalCount[e.Item1] + ")")));

            File.WriteAllText(outputFile, sb.ToString());
        }

        public AbstractDataSet ExtractDataSet(Func<CancelDatabaseEntry, bool> accept, ResizeStrategyEnum resizeStrategy)
        {
            var categoryNameToCount = new ConcurrentDictionary<string,int>();
            var elementIdToPaths = new List<List<string>>();
            var elementIdToDescription = new List<string>();
            var elementIdToCategoryIndex = new List<int>();
            var entries = _database.Values.Where(e => !e.IsRemoved && accept(e))
                .OrderBy(e => e.SHA1).ToArray();

            while (elementIdToPaths.Count < entries.Length)
            {
                elementIdToPaths.Add(null);
                elementIdToDescription.Add(null);
                elementIdToCategoryIndex.Add(-1);
            }

            var yExpected = new CpuTensor<float>(new [] {entries.Length, Hierarchy.RootPrediction().Length});
            void Process(int elementId)
            {
                var entry = entries[elementId];
                var categoryPath = ToPath(entry.Cancel);
                //Debug.Assert(categoryPath != null && categoryPath.Length >= 1);
                var categoryName = Hierarchy.CategoryPathToCategoryName(categoryPath);
                if (categoryNameToCount.ContainsKey(categoryName))
                {
                    ++categoryNameToCount[categoryName];
                }
                else
                {
                    categoryNameToCount[categoryName] = 1;
                }
                elementIdToPaths[elementId] = new List<string> {entry.Path(_rootPath)};
                elementIdToDescription[elementId] = entry.SHA1;
                elementIdToCategoryIndex[elementId] = -1;
                var elementPrediction = Hierarchy.ExpectedPrediction(categoryPath);
                Debug.Assert(elementPrediction != null);
                Debug.Assert(elementPrediction.Length == yExpected.Shape[1]);
                for (int col = 0; col < elementPrediction.Length; ++col)
                {
                    yExpected.Set(elementId, col, elementPrediction[col]);
                }
            }
            Parallel.For(0, entries.Length, Process);

            Log.Debug("found "+ elementIdToDescription.Count+" elements");
            //Log.Info(string.Join(Environment.NewLine, categoryNameToCount.OrderBy(e=>e.Key).Select(e=>e.Key +" : "+e.Value)));
            var categoryDescription = Enumerable.Range(0, yExpected.Shape[1]).Select(i=>i.ToString()).ToArray();
            return new DirectoryDataSet(
                elementIdToPaths, 
                elementIdToDescription, 
                elementIdToCategoryIndex, 
                yExpected, 
                "Cancel", 
                Objective_enum.Classification,
                3, 
                categoryDescription,
                CancelMeanAndVolatilityForEachChannel,
                resizeStrategy,
                null);
        }
        
        public static readonly List<Tuple<float, float>> CancelMeanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(147.02734f, 60.003986f), Tuple.Create(141.81636f, 51.15815f), Tuple.Create(130.15608f, 48.55502f) };


        /// <summary>
        /// load all images in 'path' into the database
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
                    var entry = new CancelDatabaseEntry
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
                    var destDirectory = Path.GetDirectoryName(destFileName)??"";
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
        public void CreateIDM(string idmPath, Func<CancelDatabaseEntry, bool> accept)
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
        private void LoadDatabase()
        {
            Log.Info("Loading database "+_rootPath);
            _database.Clear();
            var lines = File.ReadAllLines(CsvPath).Skip(1).ToArray();
            var entries = new CancelDatabaseEntry[lines.Length];
            void LoadEntry(int entryIdx)
            {
                var splitted = lines[entryIdx].Split(';');
                Debug.Assert(splitted.Length == 15);
                int columnIdx = 0;
                var entry = new CancelDatabaseEntry
                {
                    SHA1 = splitted[columnIdx++],
                    OriginalPath = splitted[columnIdx++],
                    FileExtension = splitted[columnIdx++],
                    Width = int.Parse(splitted[columnIdx++]),
                    Height = int.Parse(splitted[columnIdx++]),
                    AverageColor= CancelDatabaseEntry.StringToAverageColor(splitted[columnIdx++]),
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
                if (!string.IsNullOrEmpty(e.Value.SuggestedId))
                {
                    ++withSuggestedId;
                }
                if (!string.IsNullOrEmpty(e.Value.Cancel))
                {
                    ++withCancel;
                }
                if (!string.IsNullOrEmpty(e.Value.SuggestedCancel))
                {
                    ++withSuggestedCancel;
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

        private static CategoryHierarchy ComputeRootNode()
        {
            var root = CategoryHierarchy.NewRoot("", "");
            root.Add("mint");

            var used = root.Add("used", "");
            var used_star = used.Add("star", "etoile");
            used_star.AddAllNumbersWithSameNumberOfDigits("2digits", "", 39);
            used_star.Add("full", "_pleine");
            used_star.Add("empty", "_evidee");
            used_star.AddAllNumbersWithSameNumberOfDigits("1digit", "", 9);
            var used_gc = used.Add("gc");
            used_gc.AddAllNumbersWithSameNumberOfDigits("4digits", "", 6999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("3digits", "", 999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("2digits", "", 99);
            used_gc.AddAllNumbersWithSameNumberOfDigits("1digit", "", 9);
            var used_pc = used.Add("pc");
            used_pc.AddAllNumbersWithSameNumberOfDigits("4digits", "", 4999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("3digits", "", 999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("2digits", "", 99);
            used_pc.AddAllNumbersWithSameNumberOfDigits("1digit", "", 9);
            used.Add("preo1893");
            used.Add("ir");
            used.Add("cad");
            used.Add("cad_perle");
            used.Add("cad_octo");
            used.Add("cad_ondule");
            used.Add("cad_imprime", "imprime_rouge");
            used.Add("cad_passe", "passe");
            used.Add("cad_barcelona", "barcelona");
            used.Add("amb");
            used.Add("anchor", "ancre");
            used.Add("typo");
            used.Add("grille");
            used.Add("grille_ss_fin");
            used.Add("gros_points");
            used.Add("asna");
            used.Add("plume");

            return root;
        }


    }
}