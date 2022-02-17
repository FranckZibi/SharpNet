using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;
using SharpNet.MathTools;
using SharpNet.Models;
using SharpNet.Networks;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

namespace SharpNet.Datasets.CFM60
{
    public class CFM60DataSet : AbstractDataSet, ITimeSeriesDataSet
    {
        public CFM60HyperParameters Sample => _cfm60NetworkSample.CFM60HyperParameters;
        private readonly Cfm60NetworkSample _cfm60NetworkSample;
        private readonly CFM60DataSet TrainingDataSetIfAny;
        //TODO : add element for each valid day, even if no entry is associated with that day
        private readonly IDictionary<int, List<CFM60Entry>> _pidToSortedEntries = new Dictionary<int, List<CFM60Entry>>();
        private readonly IDictionary<int, int> _CFM60EntryIDToIndexIn_pidToSortedEntries = new Dictionary<int, int>();
        private readonly IDictionary<int, LinearRegression> PidToLinearRegressionBetweenDayAndY;

        ///For each featureId, a Tuple with:
        ///     Item1:  feature minimum
        ///     Item2:  feature maximum
        ///     Item3:  feature mean
        ///     Item4:  feature volatility
        ///     Item5:  feature correlation with label
        ///     Item6:  feature importances
        private readonly Tuple<double, double, double, double, double, double>[] Encoder_FeaturesStatistics;
        private readonly Tuple<double, double, double, double, double, double>[] Decoder_FeaturesStatistics;

        private readonly IDictionary<int, CFM60Entry> _elementIdToLastAssociateCFM60Entry = new Dictionary<int, CFM60Entry>();

        // CFM60EntryID = CFM60Entry.ID: the unique ID of a CFM60Entry
        // elementId : id of an element in the dataSet (in range [0, dataSet.Count[ )
        private readonly IDictionary<int, float> _idToPrediction = new Dictionary<int, float>();

        /// <summary>
        /// day is the end of year
        /// </summary>
        private static readonly int[] SortedEndOfYear = { 19, 269, 519, 770, 1021 };
        public static readonly HashSet<int> EndOfYear = new HashSet<int>(SortedEndOfYear);
        /// <summary>
        /// day is the end of teh trimester
        /// </summary>
        public static readonly HashSet<int> EndOfTrimester = new HashSet<int>(new[]
                                                                           {
                                                                                    77, 327, 577, 828,1084, //march
                                                                                   141, 391, 641, 892,1147, //june
                                                                                   201, 451, 702, 953, //sept
                                                                               19, 269, 519, 770,1021 // dec //TODO: check without line
                                                                           });

        /// <summary>
        /// return the day as a fraction of the day in year in ]0,1] range
        /// 1-jan   => 1/250f
        /// 31-dec  => 1
        /// </summary>
        /// <param name="day"></param>
        /// <returns></returns>
        public static float DayToFractionOfYear(int day)
        {
            for (int i = SortedEndOfYear.Length - 1; i >= 0; --i)
            {
                if (day > SortedEndOfYear[i])
                {
                    float daysInYear = (i== SortedEndOfYear.Length - 1)?250f:(SortedEndOfYear[i+1]- SortedEndOfYear[i]);
                    return ((day - SortedEndOfYear[i]) / daysInYear);
                }
            }
            return (day + (250f - SortedEndOfYear[0])) / 250f;
        }


        /// <summary>
        /// day is just before the end of year (Christmas ?)
        /// </summary>
        public static readonly HashSet<int> Christmas = new HashSet<int>(new[] { 14,264,514,765,1016});

        #region private fields

        [NotNull] public readonly CFM60Entry[] Entries;

        // ReSharper disable once NotAccessedField.Local
        private const float ls_min = -6.984719276428223f;
        private const float ls_max = 2.7631983757019043f;
        private const float nlv_min = -4.3548359870910645f;
        private const float nlv_max = 4.443584442138672f;

        ///// <summary>
        ///// Statistics of the expected outcome 'y' of the all dataSet (E(y), Vol(y), Max, Min)
        ///// </summary>
        //private readonly DoubleAccumulator Y_stats = new DoubleAccumulator();

        #endregion

        public CFM60DataSet(string xFile, string yFileIfAny, Action<string> log, Cfm60NetworkSample sample, CFM60DataSet trainingDataSetIfAny = null) 
            : this(CFM60Entry.Load(xFile, yFileIfAny, log, sample.CFM60HyperParameters.ValueToPredict, sample.CFM60HyperParameters.predictionFilesIfComputeErrors), sample, trainingDataSetIfAny)
        {
        }
        
        public CFM60DataSet(CFM60Entry[] entries, Cfm60NetworkSample cfm60NetworkSample, CFM60DataSet trainingDataSetIfAny = null)
            : base(cfm60NetworkSample.CFM60HyperParameters.IsTryingToPredictErrors? "CFM60Errors":"CFM60", 
                Objective_enum.Regression,
                cfm60NetworkSample.CFM60HyperParameters.Encoder_TimeSteps,
                new[] {"NONE"},
                null,
                ResizeStrategyEnum.None,
                cfm60NetworkSample.CFM60HyperParameters.ComputeFeatureNames(),
                new string[0],
                UseBackgroundThreadToLoadNextMiniBatch(trainingDataSetIfAny))
        {
            _cfm60NetworkSample = cfm60NetworkSample;
            TrainingDataSetIfAny = trainingDataSetIfAny;
            Entries = entries;
            int elementId = 0;

            //we initialize: _pidToSortedEntries
            foreach (var entry in Entries.OrderBy(e => e.pid).ThenBy(e => e.day))
            {
                if (!_pidToSortedEntries.ContainsKey(entry.pid))
                {
                    _pidToSortedEntries[entry.pid] = new List<CFM60Entry>();
                }

                _pidToSortedEntries[entry.pid].Add(entry);
            }

            if (IsValidationOrTestDataSet && EntriesCountForEachElementId_Y>1)
            {
                // we need to make sure that each pid has a multiple of 'EntriesCountForEachElementId_Y' as number of entries
                foreach (var l in _pidToSortedEntries.Values)
                {
                    while (l.Count % EntriesCountForEachElementId_Y != 0)
                    {
                        l.Add(CFM60Entry.Interpolate(l.Last(), null, l.Last().day+1));
                    }
                }
            }

            //we initialize _IDToIndexIn_pidToSortedEntries
            foreach (var e in _pidToSortedEntries.Values)
            {
                for (int index_in_pidToSortedEntries = 0;index_in_pidToSortedEntries < e.Count;++index_in_pidToSortedEntries)
                {
                    _CFM60EntryIDToIndexIn_pidToSortedEntries[e[index_in_pidToSortedEntries].ID] = index_in_pidToSortedEntries;
                }
            }

            //we initialize _elementIdToAssociateLastCFM60Entry
            int longestEntry = _pidToSortedEntries.Values.Select(x => x.Count).Max();
            int[] pids = _pidToSortedEntries.Keys.OrderBy(x => x).ToArray();
            int idxLastEntry = IsTrainingDataSet
                ? EntriesCountForEachElementId_X+EntriesCountForEachElementId_Y - 1
                : EntriesCountForEachElementId_Y - 1;
            while(idxLastEntry < longestEntry)
            {
                foreach (var pid in pids)
                {
                    var pidEntries = _pidToSortedEntries[pid];
                    if (idxLastEntry < pidEntries.Count)
                    {
                        _elementIdToLastAssociateCFM60Entry[elementId] = pidEntries[idxLastEntry];
                        ++elementId;
                    }
                }

                if (IsTrainingDataSet)
                {
                    //in the Training DataSet: only entries in the range [TimeSteps, +infinite[ can be trained
                    idxLastEntry += 1;
                }
                else
                {
                    //in the Validation/Test DataSets: each element is a prediction to make
                    idxLastEntry += EntriesCountForEachElementId_Y; 
                }
            }

            //we initialize Y
            //total number of items in the dataSet
            int count = elementId;
            var yData = new float[count * EntriesCountForEachElementId_Y];
            int nextIdxInY = 0;
            for (elementId = 0; elementId < count; ++elementId)
            {
                foreach(var e in ElementId_to_YEntries(elementId))
                {
                    yData[nextIdxInY++] = e.Y;
                }
            }
            Debug.Assert(nextIdxInY == yData.Length);
            Y = new CpuTensor<float>(new[] { count, EntriesCountForEachElementId_Y }, yData);

            //if we are in a training data set
            if (trainingDataSetIfAny == null) 
            {
                //we ensure that the training data set is valid
                foreach (var (pid, trainingEntries) in _pidToSortedEntries)
                {
                    if (trainingEntries.Count <= Total_TimeSteps)
                    {
                        throw new Exception("invalid Training DataSet: not enough entries (" + trainingEntries.Count + ") for pid " + pid);
                    }

                    if (trainingEntries.Any(x => double.IsNaN(x.Y)))
                    {
                        throw new Exception("invalid Training DataSet: no known Y value for pid " + pid);
                    }
                }
                Encoder_FeaturesStatistics = Sample.Compute_Encoder_FeaturesStatistics();
                Decoder_FeaturesStatistics = Sample.Compute_Decoder_FeaturesStatistics();
                PidToLinearRegressionBetweenDayAndY = ComputePidToLinearRegressionBetweenDayAndY();
            }
            else //validation or test data set
            {
                //we ensure that the associate training data set is valid
                foreach (var (pid, validationEntries) in _pidToSortedEntries)
                {
                    if (!trainingDataSetIfAny._pidToSortedEntries.ContainsKey(pid))
                    {
                        throw new Exception("validation pid "+ pid + " doesn't exist in training data set");
                    }
                    if (validationEntries.Count == 0)
                    {
                        throw new Exception("not enough entries (" + validationEntries.Count + ") in validation data set for pid " + pid);
                    }
                }
                Encoder_FeaturesStatistics = trainingDataSetIfAny.Encoder_FeaturesStatistics;
                Decoder_FeaturesStatistics = trainingDataSetIfAny.Decoder_FeaturesStatistics;
                PidToLinearRegressionBetweenDayAndY = null;
            }

        }
        // ReSharper disable once UnusedMember.Global
        public void ComputeFeatureImportance(string filePath, bool computeExtraFeature)
        { 
            var calculator = new FeatureImportancesCalculator(computeExtraFeature);
            foreach(var (_,entries) in _pidToSortedEntries)
            { 
                for (var index = 0; index < entries.Count; index++)
                {
                    var entry = entries[index];
                    var previousEntry = index==0? entry: entries[index-1];
                    Debug.Assert(!double.IsNaN(entry.Y));
                    calculator.AddFeature(previousEntry.Y, "prev_y");
                    calculator.AddFeature(entry.rel_vol, "rel_vol");
                    calculator.AddFeature(entry.abs_ret, "abs_ret");
                    //acc.AddFeature(entry.Get_mean_abs_ret(), "mean_abs_ret");
                    //acc.AddFeature((entry.Get_mean_abs_ret() - 0.118588544f) / 0.08134923f, "mean(abs_ret_normalized)");
                    calculator.AddFeature(LinearRegressionEstimate(entry.pid, entry.day),"y_LinearRegressionEstimate");
                    calculator.AddFeature(Y_Mean(entry.pid), "mean(pid_y)");
                    calculator.AddFeature(Y_Volatility(entry.pid), "vol(pid_y)");
                    calculator.AddFeature(Y_Variance(entry.pid), "var(pid_y)");
                    calculator.AddFeature(entry.Get_rel_vol_CoefficientOfVariation(), "rel_vol_CoefficientOfVariation");
                    calculator.AddFeature(entry.Get_volatility_rel_vol(), "vol(rel_vol)");
                    calculator.AddFeature(entry.LS, "LS");
                    calculator.AddFeature(CFM60Utils.NormalizeBetween_0_and_1(entry.LS, ls_min, ls_max), "NormalizeLS");
                    calculator.AddFeature((entry.LS + 3.185075f) / 1.072115f, "NormalizeLS_V2");
                    calculator.AddFeature(entry.NLV, "NLV");
                    calculator.AddFeature(CFM60Utils.NormalizeBetween_0_and_1(entry.NLV, nlv_min, nlv_max), "NormalizeNLV");
                    calculator.AddFeature((entry.NLV + 0.018128f) / 1.002737f, "NormalizeNLV_V2");
                    calculator.AddFeature(entry.day, "day");
                    calculator.AddFeature(entry.day / 250f, "day/250");
                    calculator.AddFeature(entry.day / 1151f, "day/1151");
                    calculator.AddFeature(DayToFractionOfYear(entry.day), "fraction_of_year");
                    calculator.AddFeature(EndOfYear.Contains(entry.day) ? 1 : 0, "EndOfYear_flag");
                    calculator.AddFeature(Christmas.Contains(entry.day) ? 1 : 0, "Christmas_flag");
                    calculator.AddFeature(EndOfTrimester.Contains(entry.day) ? 1 : 0, "EndOfTrimester_flag");
                    //acc.AddFeature((float) r.NextDouble(), "random_0_1");
                    //acc.AddFeature(r.Next(0, 2), "random_0_or_1");
                    //acc.AddFeature((float) (2 * r.NextDouble() - 1), "random_min1_1");
                    //acc.AddFeature(0f, "const_0");
                    //acc.AddFeature(1f, "const_1");
                    calculator.AddLabel(entry.Y);
                }
            }
            calculator.Write(filePath);
        }

        //public int Encoder_TimeSteps => Cfm60NetworkBuilder.Encoder_TimeSteps;
        public int Total_TimeSteps => Sample.Total_TimeSteps;

        int EntriesCountForEachElementId_X =>
            Sample.Use_Decoder
                ? Sample.Encoder_TimeSteps: 
                1+ Sample.Encoder_TimeSteps;

        int EntriesCountForEachElementId_Y => Sample.Use_Decoder ? Sample.Decoder_TimeSteps : 1;

        public IEnumerable<CFM60Entry> ElementId_to_YEntries(int elementId)
        {
            var lastEntry = _elementIdToLastAssociateCFM60Entry[elementId];
            var pidEntries = _pidToSortedEntries[lastEntry.pid];
            var lastIdx = _CFM60EntryIDToIndexIn_pidToSortedEntries[lastEntry.ID];
            int firstIdx = lastIdx - EntriesCountForEachElementId_Y + 1;
            for (int idx = firstIdx; idx<=lastIdx; ++idx)
            {
                yield return pidEntries[idx];
            }
        }

        public void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions)
        {
            Debug.Assert(batchPredictions.Count == batchElementIds.Length* EntriesCountForEachElementId_Y);
            var predictions = batchPredictions.ContentAsFloatArray();
            int nextPredictionIdx = 0;
            foreach (var elementId in batchElementIds)
            {
                foreach (var e in ElementId_to_YEntries(elementId))
                {
                    _idToPrediction[e.ID] = predictions[nextPredictionIdx++];
                }
            }
            Debug.Assert(nextPredictionIdx == batchPredictions.Count);
        }


        /// <summary>
        /// the sub part of the original (and complete) Training Data Set used for Validation
        /// </summary>
        public CFM60DataSet ValidationDataSet { get; set; }

        /// <summary>
        /// the original (and complete) Test Data Set for the CFM60 challenge
        /// </summary>
        public CFM60DataSet OriginalTestDataSet { get; set; }
        public Tuple<double, double, double, double, double, double> GetEncoderFeatureStatistics(int featureId)
        {
            return Encoder_FeaturesStatistics[featureId];
        }

        public IDictionary<int, LinearRegression> ComputePidToLinearRegressionBetweenDayAndY()
        {
            if (IsValidationOrTestDataSet)
            {
                //We do not use Y in validation / test data set
                return null;
            }

            return CFM60Utils.ComputePidToLinearRegressionBetweenDayAndY(Entries);
        }

        /// <summary>
        /// for Neural network model:
        ///     will save also the pid features, and the prediction file for the Train + Validation + Test datasets
        /// </summary>
        public override void Save(IModel model, string workingDirectory, string modelName)
        {
            base.Save(model, workingDirectory, modelName);
            CreatePredictionFile(model, "_predict_train_");
            ValidationDataSet?.CreatePredictionFile(model, "_predict_valid_");
            OriginalTestDataSet?.CreatePredictionFile(model, "_predict_test_");

            if (model is Network network)
            {

                var embeddingLayer = network.Layers.FirstOrDefault(l => l is EmbeddingLayer);
                if (embeddingLayer == null)
                {
                    return;
                }
                var cpuTensor = embeddingLayer.Weights.ToCpuFloat();
                cpuTensor.Save(Path.Combine(network.WorkingDirectory, "pid_features_" + network.DynamicModelName + ".csv"),
                    row => row>=1&&row<=CFM60Entry.DISTINCT_PID_COUNT, //the first row is not used in word embedding
                    true,
                    "pid;"+string.Join(";", Enumerable.Range(0,cpuTensor.Shape[0]).Select(i=>"feature_"+i))
                    );
                return;
            }
            throw new ArgumentException($"cant' save model of type {model.GetType()}");
        }

        public void CreatePredictionFile(IModel model, string fileSuffix)
        {
            var res = model.Predict(this);
            var CFM60EntryIDToPrediction = new Dictionary<int, double>();
            var spanResult = res.ReadonlyContent;
            for (int elementId = 0; elementId < Count; ++elementId)
            {
                var id = _elementIdToLastAssociateCFM60Entry[elementId].ID;
                var prediction = spanResult[elementId];
                if (Sample.ValueToPredict == CFM60HyperParameters.ValueToPredictEnum.Y_TRUE_MINUS_LR)
                {
                    prediction += CFM60Utils.LinearRegressionEstimate(id);
                }
                else if (Sample.ValueToPredict == CFM60HyperParameters.ValueToPredictEnum.Y_TRUE_MINUS_ADJUSTED_LR)
                {
                    prediction += CFM60Utils.LinearRegressionAdjustedByMeanEstimate(id);
                }
                CFM60EntryIDToPrediction[id] = prediction;
            }
            string filePath = Path.Combine(model.WorkingDirectory, model.ModelName + fileSuffix + ".csv");
            CFM60Utils.SavePredictions(CFM60EntryIDToPrediction, filePath);
        }

        /// <summary>
        /// Use Ensemble Learning to create the average predictions from different networks
        /// </summary>
        /// <param name="directory">the directory where the predictions files are located</param>
        /// <param name="fileNameWithPredictionToWeight">the fileNames of the prediction files in directory 'directory' and associated weight</param>
        /// <param name="multiplierCorrection"></param>
        /// <param name="addCorrectionStart"></param>
        /// <param name="addCorrectionEnd"></param>
        /// <returns>a path to a prediction file with the weighted average of predictions</returns>
        // ReSharper disable once UnusedMember.Global
        public static string EnsembleLearning(string directory, IDictionary<string,double> fileNameWithPredictionToWeight, double multiplierCorrection = 1.0, double addCorrectionStart = 0.0, double addCorrectionEnd = double.NaN)
        {
            if (double.IsNaN(addCorrectionEnd))
            {
                addCorrectionEnd = addCorrectionStart;
            }
            if (fileNameWithPredictionToWeight == null)
            {
                fileNameWithPredictionToWeight = new Dictionary<string, double>();
                new DirectoryInfo(directory).GetFiles("*.csv").ToList().ForEach(f => fileNameWithPredictionToWeight[f.FullName] = 1);
            }

            int? predictionsByFile = null;
            var ensembleLearningPredictions = new Dictionary<int, double>();
            var totalWeights = fileNameWithPredictionToWeight.Values.Sum();

            foreach (var (fileNameWithPrediction, weight) in fileNameWithPredictionToWeight)
            {
                Console.WriteLine("Processing file "+ fileNameWithPrediction+" with weight "+weight);
                var singleFilePredictions = CFM60Utils.LoadPredictions(Path.Combine(directory, fileNameWithPrediction));
                if (!predictionsByFile.HasValue)
                {
                    predictionsByFile = singleFilePredictions.Count;
                }
                if (predictionsByFile.Value != singleFilePredictions.Count)
                {
                    throw new ArgumentException("all predictions files do not have the same number of predictions");
                }
                foreach (var (id, prediction) in singleFilePredictions)
                {
                    if (!ensembleLearningPredictions.ContainsKey(id))
                    {
                        ensembleLearningPredictions[id] = 0;
                    }
                    ensembleLearningPredictions[id] += (weight / totalWeights) * prediction;
                }
            }
            if (predictionsByFile.HasValue && predictionsByFile.Value != ensembleLearningPredictions.Count)
            {
                throw new ArgumentException("all predictions files do not have the same ID for predictions");
            }

            var ensembleLearningPredictionFile = Path.Combine(directory, "EnsembleLearning_" + DateTime.Now.Ticks + ".csv");
            CFM60Utils.SavePredictions(ensembleLearningPredictions, ensembleLearningPredictionFile, multiplierCorrection, addCorrectionStart, addCorrectionEnd);
            return ensembleLearningPredictionFile;
        }

        /// <summary>
        /// return the entry associated with the pid 'pid' for index 'indexInPidEntryArray'
        /// </summary>
        /// <param name="pid">the pid to retrieve</param>
        /// <param name="indexInPidEntryArray">
        /// index of the entry to retrieve.
        /// if strictly less then 0:
        ///     it means that we are in a validation DataSet and we want to load data from
        ///     the associated Training DataSet
        /// </param>
        /// <returns></returns>
        private CFM60Entry GetEntry(int pid, int indexInPidEntryArray)
        {
            if (indexInPidEntryArray >= 0)
            {
                return _pidToSortedEntries[pid][indexInPidEntryArray];
            }
            if (IsTrainingDataSet)
            {
                throw new Exception("no Entry for pid " + pid + " at index=" + indexInPidEntryArray);
            }
            var trainingEntries = TrainingDataSetIfAny._pidToSortedEntries[pid];
            return trainingEntries[trainingEntries.Count + indexInPidEntryArray];
        }

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            throw new Exception("should never be called");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="elementId">gives the entry to predict</param>
        /// <param name="indexInBuffer"></param>
        /// <param name="all_xBuffer">
        /// input shape: (batchSize, TimeSteps, InputSize)</param>
        /// <param name="yBuffer">
        /// output shape: (batchSize, 1)
        /// </param>
        /// <param name="withDataAugmentation"></param>
        protected override void LoadAt(int elementId, int indexInBuffer, List<CpuTensor<float>> all_xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            var xEncoder = all_xBuffer[0];
            Debug.Assert(xEncoder.Shape.Length == 3);
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < all_xBuffer[0].Shape[0]);
            Debug.Assert(xEncoder.Shape[1] == Sample.Encoder_TimeSteps);
            Debug.Assert(xEncoder.Shape[2] == Sample.Encoder_InputSize);
            Debug.Assert(xEncoder.Shape[2] == Encoder_FeaturesStatistics.Length);
            Debug.Assert(yBuffer == null || all_xBuffer[0].Shape[0] == yBuffer.Shape[0]); //same batch size
            Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));
            Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));

            LoadAt(elementId, indexInBuffer, xEncoder, true);
            if (Sample.Use_Decoder)
            {
                Debug.Assert(all_xBuffer.Count == 2);
                var xDecoder = all_xBuffer[1];
                LoadAt(elementId, indexInBuffer, xDecoder, false);
            }

            if (yBuffer != null)
            {
                Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        private void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> x, bool isEncoder)
        {
            var xElementId = x.AsFloatCpuSpan.Slice(indexInBuffer * x.MultDim0, x.MultDim0);

            int idx = 0;
            var lastAssociateCFM60Entry = _elementIdToLastAssociateCFM60Entry[elementId];
            var pid = lastAssociateCFM60Entry.pid;
            int lastIndexInPidEntries = _CFM60EntryIDToIndexIn_pidToSortedEntries[lastAssociateCFM60Entry.ID];
            Debug.Assert(_pidToSortedEntries[pid][lastIndexInPidEntries].ID == lastAssociateCFM60Entry.ID);
            if (isEncoder && Sample.Use_Decoder)
            {
                Debug.Assert(Sample.Decoder_TimeSteps >= 1);
                lastIndexInPidEntries -= Sample.Decoder_TimeSteps;
            }

            int timeSteps = x.Shape[1];
            int featuresLength = x.Shape[2];
            // ReSharper disable once LoopVariableIsNeverChangedInsideLoop
            for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
            {
                var indexInPidEntryArray = lastIndexInPidEntries - timeSteps + timeStep + 1;
                var entry = GetEntry(pid, indexInPidEntryArray);

                //pid
                if (Sample.Pid_EmbeddingDim >= 1)
                {
                    //pids are in range  [0, 899]
                    //EmbeddingLayer is expecting them in range [1,900] that's why we add +1
                    xElementId[idx++] = entry.pid + 1;
                }
               
                if (Sample.Use_y_LinearRegressionEstimate)
                {
                    xElementId[idx++] = Normalize(LinearRegressionEstimate(entry.pid, entry.day), idx % featuresLength, isEncoder);
                }
                if (Sample.Use_mean_pid_y)
                {
                    xElementId[idx++] = Normalize(Y_Mean(entry.pid), idx % featuresLength, isEncoder);
                }
                if (Sample.Use_volatility_pid_y)
                {
                    xElementId[idx++] = Normalize(Y_Volatility(entry.pid), idx % featuresLength, isEncoder);
                }
                if (Sample.Use_variance_pid_y)
                {
                    xElementId[idx++] = Normalize(Y_Variance(entry.pid), idx % featuresLength, isEncoder);
                }
                //day/year
                if (Sample.Use_day)
                {
                    xElementId[idx++] = Normalize( entry.day / Sample.Use_day_Divider, idx % featuresLength, isEncoder);
                }
                if (Sample.Use_fraction_of_year)
                {
                    xElementId[idx++] = Normalize(DayToFractionOfYear(entry.day), idx % featuresLength, isEncoder);
                }
                if (Sample.Use_year_Cyclical_Encoding)
                {
                    xElementId[idx++] = (float)Math.Sin(2*Math.PI*DayToFractionOfYear(entry.day));
                    xElementId[idx++] = (float)Math.Cos(2*Math.PI*DayToFractionOfYear(entry.day));
                }
                if (Sample.Use_EndOfYear_flag)
                {
                    xElementId[idx++] = EndOfYear.Contains(entry.day) ? 1 : 0;
                }
                if (Sample.Use_Christmas_flag)
                {
                    xElementId[idx++] = Christmas.Contains(entry.day) ? 1 : 0;
                }
                if (Sample.Use_EndOfTrimester_flag)
                {
                    xElementId[idx++] = EndOfTrimester.Contains(entry.day) ? 1 : 0;
                }
                //abs_ret
                if (Sample.Use_abs_ret)
                {
                    //entry.abs_ret.AsSpan().CopyTo(xDest.Slice(idx, entry.abs_ret.Length));
                    //idx += entry.abs_ret.Length;
                    for (int i = 0; i < entry.abs_ret.Length; ++i)
                    {
                        xElementId[idx++] = Normalize(entry.abs_ret[i], idx % featuresLength, isEncoder);
                    }
                }
                if (Sample.Use_mean_abs_ret)
                {
                    xElementId[idx++] = Normalize(entry.Get_mean_abs_ret(), idx % featuresLength, isEncoder);
                }
                if (Sample.Use_volatility_abs_ret)
                {
                    xElementId[idx++] = Normalize(entry.Get_volatility_abs_ret(), idx % featuresLength, isEncoder);
                }
                //rel_vol
                if (Sample.Use_rel_vol)
                {
                    //var asSpan = entry.rel_vol.AsSpan();
                    if (Sample.Use_rel_vol_start_and_end_only)
                    {
                        //asSpan.Slice(0, 12).CopyTo(xDest.Slice(idx, 12));
                        //asSpan.Slice(entry.rel_vol.Length - 12, 12).CopyTo(xDest.Slice(idx + 12, 12));
                        //idx += 2 * 12;
                        for (int i = 0; i < 12; ++i)
                        {
                            xElementId[idx++] = Normalize(entry.rel_vol[i], idx % featuresLength, isEncoder);
                        }

                        for (int i = entry.rel_vol.Length - 12; i < entry.rel_vol.Length; ++i)
                        {
                            xElementId[idx++] = Normalize(entry.rel_vol[i], idx % featuresLength, isEncoder);
                        }
                    }
                    else
                    {
                        //asSpan.CopyTo(xDest.Slice(idx, entry.rel_vol.Length));
                        //idx += entry.rel_vol.Length;
                        for (int i = 0; i < entry.rel_vol.Length; ++i)
                        {
                            xElementId[idx++] = Normalize(entry.rel_vol[i], idx % featuresLength, isEncoder);
                        }
                    }
                }
                if (Sample.Use_volatility_rel_vol)
                {
                    xElementId[idx++] = Normalize(entry.Get_volatility_rel_vol(), idx % featuresLength, isEncoder);
                }
                //LS
                if (Sample.Use_LS)
                {
                    xElementId[idx++] = Normalize(entry.LS, idx % featuresLength, isEncoder);
                }
                //NLV
                if (Sample.Use_NLV)
                {
                    xElementId[idx++] = Normalize(entry.NLV, idx % featuresLength, isEncoder);
                }
                //y estimate
                if (Sample.Use_prev_Y && isEncoder)
                {
                    var indexOfyEntryInPidEntryArray = Sample.Use_Decoder
                        ? indexInPidEntryArray
                        : indexInPidEntryArray - 1;  //we take the previous entry
                    var yEntry = GetEntry(pid, indexOfyEntryInPidEntryArray);
                    if (IsTrainingDataSet
                        || indexOfyEntryInPidEntryArray < 0 //the entry is in the training set
                    )
                    {
                        //we will use the true value for Y
                        var y = yEntry.Y;
                        if (double.IsNaN(y))
                        {
                            throw new Exception("no Y value associated with entry " + (indexInPidEntryArray - 1) + " of pid " + pid);
                        }
                        // ReSharper disable once ConditionIsAlwaysTrueOrFalse
                        xElementId[idx++] = Normalize(y, idx % featuresLength, isEncoder);
                    }
                    else
                    {
                        //we need to use the estimated value for Y (even if the true value of Y is available)
                        if (!_idToPrediction.ContainsKey(yEntry.ID))
                        {
                            throw new Exception("missing prediction for ID " + yEntry.ID + " with pid " + pid + " : it is required to make the prediction for next ID " + entry.ID);
                        }
                        // ReSharper disable once ConditionIsAlwaysTrueOrFalse
                        xElementId[idx++] = Normalize(_idToPrediction[yEntry.ID], idx % featuresLength, isEncoder);
                    }
                }

                int expectedInputSize = isEncoder ? Sample.Encoder_InputSize : Sample.Decoder_InputSize;
                if (timeStep == 0 && elementId == 0 && idx != expectedInputSize)
                {
                    throw new Exception("expecting " + expectedInputSize + " elements but got " + idx);
                }
            }
        }

        private float Normalize(float featureValue, int featureId, bool isEncoder)
        {
            if (TrainingDataSetIfAny != null)
            {
                //in validation/test data set, we'll normalize using the training data set stats
                return TrainingDataSetIfAny.Normalize(featureValue, featureId, isEncoder);
            }

            Debug.Assert(IsTrainingDataSet);
            var stats = isEncoder
                ?Encoder_FeaturesStatistics[featureId]
                :Decoder_FeaturesStatistics[featureId];
            if (  stats == null 
                ||Sample.InputNormalizationType == CFM60HyperParameters.InputNormalizationEnum.NONE
                ||Sample.InputNormalizationType == CFM60HyperParameters.InputNormalizationEnum.BATCH_NORM_LAYER
                )
            {
                return featureValue;
            }
            if (Sample.InputNormalizationType == CFM60HyperParameters.InputNormalizationEnum.Z_SCORE)
            {
                var mean = (float)stats.Item3;
                var volatility = (float)stats.Item4;
                return (featureValue - mean) / volatility;
            }
            if (  Sample.InputNormalizationType == CFM60HyperParameters.InputNormalizationEnum.DEDUCE_MEAN
                ||Sample.InputNormalizationType == CFM60HyperParameters.InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER)
            {
                var mean = (float)stats.Item3;
                return featureValue - mean;
            }
            throw new NotImplementedException("not supported " + Sample.InputNormalizationType);
        }

        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            var dayThreshold = CFM60Utils.DayThreshold(Entries, percentageInTrainingSet);
            var training = new CFM60DataSet(Entries.Where(e => e.day <= dayThreshold).ToArray(), _cfm60NetworkSample);
            var validation = new CFM60DataSet(Entries.Where(e => e.day > dayThreshold).ToArray(), _cfm60NetworkSample, training);
            return new TrainingAndTestDataLoader(training, validation, Name);
        }


        public override IDataSet SubDataSet(double percentageToKeep)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// we'll save the network if we have reached a very small loss
        /// </summary>
        public override bool ShouldCreateSnapshotForEpoch(int epoch, Network network)
        {
            return    epoch >= 2
                   && network.CurrentEpochIsAbsolutelyBestInValidationLoss()
                   && !double.IsNaN(network.EpochData.Last().ValidationLoss)
                   && network.Config.AlwaysUseFullTestDataSetForLossAndAccuracy
                   && network.EpochData.Last().ValidationLoss < Sample.MaxLossToSaveTheNetwork;
        }

        public override int Count => Y.Shape[0];

        public override int ElementIdToCategoryIndex(int elementId)
        {
            return -1;
        }

        public override double PercentageToUseForLossAndAccuracyFastEstimate => 0.0; //we do not compute any estimate

        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public override CpuTensor<float> Y { get; }

        public override string ToString()
        {
            var xShape = new [] {Count, Sample.Encoder_TimeSteps, Sample.Encoder_InputSize, Encoder_FeaturesStatistics.Length};
            return Tensor.ShapeToString(xShape) + " => " + Tensor.ShapeToString(Y.Shape);
        }

        private static bool UseBackgroundThreadToLoadNextMiniBatch(CFM60DataSet trainingDataSetIfAny)
        {
            if (trainingDataSetIfAny != null)
            {
                //for Validation/Test DataSet, we should not use a background thread for loading next mini batch data
                return false;
            }
            //for Training DataSet, we should use background thread for loading next mini batch
            return true;
        }

        public bool IsTrainingDataSet => TrainingDataSetIfAny == null;
        public bool IsValidationOrTestDataSet => !IsTrainingDataSet;

        public override List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer)
        {
            var result = new List<int[]> {shapeForFirstLayer};
            if (Sample.Use_Decoder)
            {
                var inputShapeDecoder = new []
                {
                    shapeForFirstLayer[0],
                    Sample.Decoder_TimeSteps,
                    Sample.Decoder_InputSize
                };
                result.Add(inputShapeDecoder);
            }
            return result;
        }


        private float LinearRegressionEstimate(int pid, int day)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Estimation(day);
        }
        private float Y_Mean(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Average;
        }
        private float Y_Volatility(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Volatility;
        }

        private float Y_Variance(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Variance;
        }


        protected override int GetMaxElementsToLoad(int[] shuffledElementId, int firstIndexInShuffledElementId, int batchSize)
        {
            var defaultResult = base.GetMaxElementsToLoad(shuffledElementId, firstIndexInShuffledElementId, batchSize);
            if (IsTrainingDataSet)
            {
                return defaultResult;
            }

            //in Validation & Test DataSet, we can only make at most 1 prediction / pid in each mini batch
            var observedPid = new HashSet<int>();
            for (int indexInShuffledElementId = firstIndexInShuffledElementId; indexInShuffledElementId < shuffledElementId.Length; ++indexInShuffledElementId)
            {
                int currentLength = indexInShuffledElementId - firstIndexInShuffledElementId;
                if (currentLength >= batchSize)
                {
                    return batchSize;
                }
                var elementId = shuffledElementId[indexInShuffledElementId];
                var pid = _elementIdToLastAssociateCFM60Entry[elementId].pid;
                if (!observedPid.Add(pid))
                {
                    //we have already see this pid before.
                    //We can only make one prediction / pid in each batch
                    return currentLength;
                }
            }
            return defaultResult;
        }
    }
}
