﻿using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.MathTools;
using SharpNet.Networks;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

/*
==========================================================================
= percentageInTrainingSet: 90 %
==========================================================================
abs_ret:
		ALL Training: E(x)=0.118589; Vol(X)=0.18531; Count=41753402; Min=0; Max=33.171913146972656
				Splitted:Training: E(x)=0.119316; Vol(X)=0.187061; Count=37592958; Min=0; Max=22.816667556762695
				Splitted:Validation: E(x)=0.112014; Vol(X)=0.168528; Count=4160444; Min=0; Max=33.171913146972656
		Test E(x)=0.129822; Vol(X)=0.233596; Count=19016384; Min=0; Max=99.36603546142578
		All: E(x)=0.122104; Vol(X)=0.201734; Count=60769786; Min=0; Max=99.36603546142578
days:
		ALL Training: E(x)=412.493759; Vol(X)=231.328505; Count=684482; Min=0; Max=804
				Splitted:Training: E(x)=373.313415; Vol(X)=209.70433; Count=616278; Min=0; Max=728
				Splitted:Validation: E(x)=766.519676; Vol(X)=21.95167; Count=68204; Min=729; Max=804
		Test E(x)=977.772153; Vol(X)=100.09615; Count=311744; Min=805; Max=1151
		All: E(x)=589.383488; Vol(X)=329.552116; Count=996226; Min=0; Max=1151
LS:
		ALL Training: E(x)=-3.185075; Vol(X)=1.072115; Count=684482; Min=-6.984719276428223; Max=2.7631983757019043
				Splitted:Training: E(x)=-3.192015; Vol(X)=1.068935; Count=616278; Min=-6.984719276428223; Max=2.7631983757019043
				Splitted:Validation: E(x)=-3.122362; Vol(X)=1.098445; Count=68204; Min=-4.605170249938965; Max=1.7952927350997925
		Test E(x)=-2.919658; Vol(X)=1.176852; Count=311744; Min=-4.605170249938965; Max=2.479624032974243
		All: E(x)=-3.102019; Vol(X)=1.112783; Count=996226; Min=-6.984719276428223; Max=2.7631983757019043
NLV:
		ALL Training: E(x)=-0.018128; Vol(X)=1.002737; Count=684482; Min=-4.3548359870910645; Max=4.443584442138672
				Splitted:Training: E(x)=-0.020225; Vol(X)=1.003731; Count=616278; Min=-4.3548359870910645; Max=4.443584442138672
				Splitted:Validation: E(x)=0.000818; Vol(X)=0.993511; Count=68204; Min=-3.3405704498291016; Max=3.7534914016723633
		Test E(x)=0.039803; Vol(X)=0.992802; Count=311744; Min=-3.3068394660949707; Max=4.196197032928467
		All: E(x)=-0; Vol(X)=0.999999; Count=996226; Min=-4.3548359870910645; Max=4.443584442138672
ret_vol:
		ALL Training: E(x)=0.016393; Vol(X)=0.018006; Count=41753402; Min=0; Max=1
				Splitted:Training: E(x)=0.016393; Vol(X)=0.018024; Count=37592958; Min=0; Max=1
				Splitted:Validation: E(x)=0.016393; Vol(X)=0.017848; Count=4160444; Min=0; Max=1
		Test E(x)=0.016393; Vol(X)=0.01777; Count=19016384; Min=0; Max=0.9172645807266235
		All: E(x)=0.016393; Vol(X)=0.017933; Count=60769786; Min=0; Max=1
Y:
		ALL Training: E(x)=-1.958691; Vol(X)=0.909245; Count=684482; Min=-7.137685775756836; Max=3.580919027328491
				Splitted:Training: E(x)=-1.987459; Vol(X)=0.908148; Count=616278; Min=-7.137685775756836; Max=3.580919027328491
				Splitted:Validation: E(x)=-1.698744; Vol(X)=0.87732; Count=68204; Min=-5.528903484344482; Max=3.087144374847412
		Test N/A
		All: E(x)=-1.958691; Vol(X)=0.909245; Count=684482; Min=-7.137685775756836; Max=3.580919027328491

==========================================================================
= percentageInTrainingSet: 68 %
==========================================================================
abs_ret:
		ALL Training: E(x)=0.118589; Vol(X)=0.18531; Count=41753402; Min=0; Max=33.171913146972656
				Splitted:Training: E(x)=0.121058; Vol(X)=0.192538; Count=28410262; Min=0; Max=22.816667556762695
				Splitted:Validation: E(x)=0.11333; Vol(X)=0.168773; Count=13343140; Min=0; Max=33.171913146972656
		Test E(x)=0.129822; Vol(X)=0.233596; Count=19016384; Min=0; Max=99.36603546142578
		All: E(x)=0.122104; Vol(X)=0.201734; Count=60769786; Min=0; Max=99.36603546142578
days:
		ALL Training: E(x)=412.493759; Vol(X)=231.328505; Count=684482; Min=0; Max=804
				Splitted:Training: E(x)=285.788256; Vol(X)=161.415105; Count=465742; Min=0; Max=558
				Splitted:Validation: E(x)=682.275574; Vol(X)=70.792443; Count=218740; Min=559; Max=804
		Test E(x)=977.772153; Vol(X)=100.09615; Count=311744; Min=805; Max=1151
		All: E(x)=589.383488; Vol(X)=329.552116; Count=996226; Min=0; Max=1151
LS:
		ALL Training: E(x)=-3.185075; Vol(X)=1.072115; Count=684482; Min=-6.984719276428223; Max=2.7631983757019043
				Splitted:Training: E(x)=-3.196913; Vol(X)=1.067255; Count=465742; Min=-6.984719276428223; Max=2.7631983757019043
				Splitted:Validation: E(x)=-3.159868; Vol(X)=1.081958; Count=218740; Min=-4.605170249938965; Max=1.9762426614761353
		Test E(x)=-2.919658; Vol(X)=1.176852; Count=311744; Min=-4.605170249938965; Max=2.479624032974243
		All: E(x)=-3.102019; Vol(X)=1.112783; Count=996226; Min=-6.984719276428223; Max=2.7631983757019043
NLV:
		ALL Training: E(x)=-0.018128; Vol(X)=1.002737; Count=684482; Min=-4.3548359870910645; Max=4.443584442138672
				Splitted:Training: E(x)=-0.016303; Vol(X)=1.006889; Count=465742; Min=-4.3548359870910645; Max=4.443584442138672
				Splitted:Validation: E(x)=-0.022014; Vol(X)=0.993827; Count=218740; Min=-3.3468830585479736; Max=3.861933708190918
		Test E(x)=0.039803; Vol(X)=0.992802; Count=311744; Min=-3.3068394660949707; Max=4.196197032928467
		All: E(x)=-0; Vol(X)=0.999999; Count=996226; Min=-4.3548359870910645; Max=4.443584442138672
ret_vol:
		ALL Training: E(x)=0.016393; Vol(X)=0.018006; Count=41753402; Min=0; Max=1
				Splitted:Training: E(x)=0.016393; Vol(X)=0.018106; Count=28410262; Min=0; Max=1
				Splitted:Validation: E(x)=0.016393; Vol(X)=0.017792; Count=13343140; Min=0; Max=1
		Test E(x)=0.016393; Vol(X)=0.01777; Count=19016384; Min=0; Max=0.9172645807266235
		All: E(x)=0.016393; Vol(X)=0.017933; Count=60769786; Min=0; Max=1
Y:
		ALL Training: E(x)=-1.958691; Vol(X)=0.909245; Count=684482; Min=-7.137685775756836; Max=3.580919027328491
				Splitted:Training: E(x)=-2.061522; Vol(X)=0.914849; Count=465742; Min=-7.137685775756836; Max=3.580919027328491
				Splitted:Validation: E(x)=-1.739741; Vol(X)=0.857033; Count=218740; Min=-5.98342752456665; Max=3.087144374847412
		Test N/A
		All: E(x)=-1.958691; Vol(X)=0.909245; Count=684482; Min=-7.137685775756836; Max=3.580919027328491
*/

namespace SharpNet.Datasets
{
    public class CFM60DataSet : AbstractDataSet, IDataSetWithExpectedAverage, ITimeSeriesDataSet
    {
        private readonly CFM60NetworkBuilder Cfm60NetworkBuilder;
        private readonly CFM60DataSet TrainingDataSetIfAny;
        private readonly IDictionary<int, List<CFM60Entry>> _pidToSortedEntries = new Dictionary<int, List<CFM60Entry>>();
        private readonly IDictionary<int, CFM60Entry> _elementIdToEntryToPredict = new Dictionary<int, CFM60Entry>();

        // CFM60EntryID = CFM60Entry.ID: the unique ID of a CFM60Entry
        // elementId : id of an element in the dataSet (in range [0, dataSet.Count[ )
        private readonly IDictionary<int, int> _CFM60EntryIDToElementId= new Dictionary<int, int>();
        private readonly IDictionary<int, float> _elementIdToPrediction = new Dictionary<int, float>();

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

        public CFM60DataSet(string xFile, string yFileIfAny, Action<string> log, CFM60NetworkBuilder cfm60NetworkBuilder, CFM60DataSet trainingDataSetIfAny = null) 
            : this(CFM60Entry.Load(xFile, yFileIfAny, log), cfm60NetworkBuilder, trainingDataSetIfAny)
        {
        }

        public CFM60DataSet(CFM60Entry[] entries, CFM60NetworkBuilder cfm60NetworkBuilder, CFM60DataSet trainingDataSetIfAny = null)
            : base("CFM60",
                cfm60NetworkBuilder.TimeSteps,
                new[] {"NONE"},
                null,
                ResizeStrategyEnum.None,
                UseBackgroundThreadToLoadNextMiniBatch(trainingDataSetIfAny))
        {
            Cfm60NetworkBuilder = cfm60NetworkBuilder;
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

            //we initialize: _elementIdToEntryToPredict and _entryIdToElementId
            int longestEntry = _pidToSortedEntries.Values.Select(x => x.Count).Max();
            int[] pids = _pidToSortedEntries.Keys.OrderBy(x => x).ToArray();
            for (int i = 0; i < longestEntry; ++i)
            {
                foreach (var pid in pids)
                {
                    var pidEntries = _pidToSortedEntries[pid];
                    if (i < pidEntries.Count)
                    {
                        if (  !IsTrainingDataSet  //in the Validation/Test DataSets: each element is a prediction to make
                            || (i >= TimeSteps)) //in the Training DataSet: only entries in the range [TimeSteps, +infinite[ can be trained
                        {
                            _elementIdToEntryToPredict[elementId] = pidEntries[i];
                            _CFM60EntryIDToElementId[pidEntries[i].ID] = elementId;
                            ++elementId;
                        }
                    }
                }
            }

            //total number of items in the dataSet
            int count = IsTrainingDataSet
                ? _pidToSortedEntries.Values.Select(e => Math.Max(e.Count - TimeSteps, 0)).Sum()
                : _pidToSortedEntries.Values.Select(e => e.Count).Sum();

            var yData = new float[count];
            for (int i = 0; i < yData.Length; ++i)
            {
                yData[i] = _elementIdToEntryToPredict[i].Y;
            }
            Y = new CpuTensor<float>(new[] { yData.Length, 1}, yData);

            if (IsTrainingDataSet)
            {
                //We ensure that the Training DataSet is valid
                foreach (var e in _pidToSortedEntries)
                {
                    if (e.Value.Count <= TimeSteps)
                    {
                        throw new Exception("invalid Training DataSet: not enough entries (" + e.Value.Count + ") for pid " + e.Key);
                    }
                    if (e.Value.Any(x => double.IsNaN(x.Y)))
                    {
                        throw new Exception("invalid Training DataSet: no known Y value for pid " + e.Key);
                    }
                }
            }
            else //Validation DataSet
            {
                //We ensure that the associate Training DataSet is valid.
                // ReSharper disable once PossibleNullReferenceException
                if (!_pidToSortedEntries.Keys.OrderBy(x => x).SequenceEqual(trainingDataSetIfAny._pidToSortedEntries.Keys.OrderBy(x => x)))
                {
                    throw new Exception("pid are incoherent between Training and Validation DataSet");
                }

                foreach (var pid in _pidToSortedEntries.Keys)
                {
                    var trainingEntries = trainingDataSetIfAny._pidToSortedEntries[pid];
                    if (trainingEntries.Count < TimeSteps)
                    {
                        throw new Exception("not enough entries (" + trainingEntries.Count + ") in Training DataSet for pid " + pid);
                    }
                }
            }
        }

        public int TimeSteps => Cfm60NetworkBuilder.TimeSteps;

        // ReSharper disable once UnusedMember.Global
        public void CreateSummaryFile(string filePath)
        {
            var sb = new StringBuilder();
            sb.Append("sep=,");
            sb.Append(Environment.NewLine + "ID,pid,day,NLV,LS,E(ret_vol),Vol(ret_vol),E(ret_vol5),Vol(ret_vol5),ret_vol_last,E(abs_ret),Vol(abs_ret),E(abs_ret5),Vol(abs_ret5),abs_ret_last,expected");
            foreach (var e in Entries)
            {
                sb.Append(Environment.NewLine + e.ID + "," + e.pid + "," + e.day + "," +e.NLV.ToString(CultureInfo.InvariantCulture) + "," + e.LS.ToString(CultureInfo.InvariantCulture));
                const int NbLast = 5;
                var retVolAccAll = new DoubleAccumulator();
                var retVolAccLast5 = new DoubleAccumulator();
                for (int i=e.ret_vol.Length-1- NbLast; i<e.ret_vol.Length;++i)
                {
                    retVolAccLast5.Add(e.ret_vol[i], 1);
                }
                foreach (var r in e.ret_vol) {retVolAccAll.Add(r,1);}
                sb.Append("," + retVolAccAll.Average + "," + retVolAccAll.Volatility);
                sb.Append("," + retVolAccLast5.Average+ "," + retVolAccLast5.Volatility);
                sb.Append("," + e.ret_vol.Last());

                var absRetAccAll = new DoubleAccumulator();
                var absRetAccLast5 = new DoubleAccumulator();
                for (int i = e.abs_ret.Length - 1 - NbLast; i < e.ret_vol.Length; ++i)
                {
                    absRetAccLast5.Add(e.abs_ret[i], 1);
                }
                foreach (var r in e.abs_ret) { absRetAccAll.Add(r, 1); }
                sb.Append("," + absRetAccAll.Average + "," + absRetAccAll.Volatility);
                sb.Append("," + absRetAccLast5.Average + "," + absRetAccLast5.Volatility);
                sb.Append("," + e.abs_ret.Last());

                if (!double.IsNaN(e.Y))
                {
                    sb.Append("," + e.Y.ToString(CultureInfo.InvariantCulture));
                }
            }
            File.WriteAllText(filePath, sb.ToString());
        }

        public IDictionary<string, DoubleAccumulator> ComputeStats()
        {
            var stats = new Dictionary<string, DoubleAccumulator>();
            var days_accumulator = new DoubleAccumulator();
            var pid_accumulator = new DoubleAccumulator();
            var LS_accumulator = new DoubleAccumulator();
            var NLV_accumulator = new DoubleAccumulator();
            var abs_ret_accumulator = new DoubleAccumulator();
            var ret_vol_accumulator = new DoubleAccumulator();



            foreach (var e in Entries)
            {
                days_accumulator.Add(e.day, 1);
                pid_accumulator.Add(e.pid, 1);
                NLV_accumulator.Add(e.NLV, 1);
                LS_accumulator.Add(e.LS, 1);
                abs_ret_accumulator.Add(e.abs_ret);
                ret_vol_accumulator.Add(e.ret_vol);
            }

            var Y_accumulator = new DoubleAccumulator();
            foreach (var y in Y.AsReadonlyFloatCpuContent)
            {
                Y_accumulator.Add(y, 1);
            }

            stats["days"] = days_accumulator;
            stats["pid"] = pid_accumulator;
            stats["LS"] = LS_accumulator;
            stats["NLV"] = NLV_accumulator;
            stats["abs_ret"] = abs_ret_accumulator;
            stats["ret_vol"] = ret_vol_accumulator;
            stats["Y"] = Y_accumulator;
            return stats;
        }

        public void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions)
        {
            Debug.Assert(batchPredictions.Count == batchElementIds.Length);
            var predictions = batchPredictions.ContentAsFloatArray();
            for (int i = 0; i < batchElementIds.Length; ++i)
            {
                _elementIdToPrediction[batchElementIds[i]] = predictions[i];
            }
        }


        public IDictionary<int, LinearRegression> ComputePidToLinearRegressionBetweenDayAndY()
        {
            var pidToLinearRegression = new Dictionary<int, LinearRegression>();
            foreach (var e in Entries)
            {
                if (!pidToLinearRegression.ContainsKey(e.pid))
                {
                    pidToLinearRegression[e.pid] = new LinearRegression();
                }
                Debug.Assert(!double.IsNaN(e.Y));
                pidToLinearRegression[e.pid].Add(e.day, e.Y);
            }
            return pidToLinearRegression;
        }

        public void CreatePredictionFile(Network network, int miniBatchSizeForAllWorkers, string filePath)
        {
            var res = network.Predict(this, miniBatchSizeForAllWorkers);
            var CFM60EntryIDToPrediction = new Dictionary<int, double>();
            var spanResult = res.ReadonlyContent;
            for (int elementId = 0; elementId < Count; ++elementId)
            {
                var entryToPredict = _elementIdToEntryToPredict[elementId];
                CFM60EntryIDToPrediction[entryToPredict.ID] = spanResult[elementId];
            }

            CreatePredictionFile(CFM60EntryIDToPrediction, filePath);
        }

        /// <summary>
        /// Use Ensemble Learning to create the average predictions from different networks
        /// </summary>
        /// <param name="directory">the directory where the predictions files are located</param>
        /// <param name="fileNameWithPredictionToWeight">the fileNames of the prediction files in directory 'directory' and associated weight</param>
        /// <returns>a path to a prediction file with the weighted average of predictions</returns>
        // ReSharper disable once UnusedMember.Global
        public static string EnsembleLearning(string directory, IDictionary<string,double> fileNameWithPredictionToWeight)
        {
            int? predictionsByFile = null;
            var ensembleLearningPredictions = new Dictionary<int, double>();
            var totalWeights = fileNameWithPredictionToWeight.Values.Sum();
            foreach (var (fileNameWithPrediction, weight) in fileNameWithPredictionToWeight)
            {
                var singleFilePredictions = LoadPredictionFile(Path.Combine(directory, fileNameWithPrediction));
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
                    ensembleLearningPredictions[id] += (weight/totalWeights) * prediction;
                }
            }
            if (predictionsByFile.HasValue && predictionsByFile.Value != ensembleLearningPredictions.Count)
            {
                throw new ArgumentException("all predictions files do not have the same ID for predictions");
            }

            var ensembleLearningPredictionFile = Path.Combine(directory, "EnsembleLearning_" + DateTime.Now.Ticks + ".csv");
            CreatePredictionFile(ensembleLearningPredictions, ensembleLearningPredictionFile);
            return ensembleLearningPredictionFile;
        }


        public static void CreatePredictionFile(IDictionary<int, double> CFM60EntryIDToPrediction, string filePath)
        {
            var sb = new StringBuilder();
            sb.Append("ID,target");
            foreach (var p in CFM60EntryIDToPrediction.OrderBy(x => x.Key))
            {
                sb.Append(Environment.NewLine + p.Key + "," + p.Value.ToString(CultureInfo.InvariantCulture));
            }

            File.WriteAllText(filePath, sb.ToString());
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


        /// <summary>
        /// 
        /// </summary>
        /// <param name="elementId">gives the entry to predict</param>
        /// <param name="indexInBuffer"></param>
        /// <param name="xBuffer">
        /// input shape: (batchSize, TimeSteps, InputSize)</param>
        /// <param name="yBuffer">
        /// output shape: (batchSize, 1)
        /// </param>
        /// <param name="withDataAugmentation"></param>
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(xBuffer.Shape.Length == 3);
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            Debug.Assert(xBuffer.Shape[1] == TimeSteps);
            Debug.Assert(xBuffer.Shape[2] == Cfm60NetworkBuilder.InputSize);
            Debug.Assert(yBuffer == null || xBuffer.Shape[0] == yBuffer.Shape[0]); //same batch size
            Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));

            var xDest = xBuffer.AsFloatCpuSpan.Slice(indexInBuffer * xBuffer.MultDim0, xBuffer.MultDim0);

            int idx = 0;
            var entryToPredict = _elementIdToEntryToPredict[elementId];
            var pid = entryToPredict.pid;
            var pidEntries = _pidToSortedEntries[pid];
            int indexInPidEntries = entryToPredict.day - pidEntries[0].day;

            for (int i = 0; i < pidEntries.Count; ++i)
            {
                if (pidEntries[i].ID == entryToPredict.ID)
                {
                    indexInPidEntries = i;
                    break;
                }
            }

            Debug.Assert(pidEntries[indexInPidEntries].ID == entryToPredict.ID);

            for (int timeStep = 0; timeStep < TimeSteps; ++timeStep)
            {
                var indexInPidEntryArray = indexInPidEntries - TimeSteps + timeStep + 1;
                var entry = GetEntry(pid, indexInPidEntryArray);

                if (Cfm60NetworkBuilder.Pid_EmbeddingDim >= 1)
                {
                    xDest[idx++] = entry.pid;
                }

                if (Cfm60NetworkBuilder.Use_prev_Y_InputTensor)
                {
                    var previousEntry = GetEntry(pid, indexInPidEntryArray - 1);
                    if (IsTrainingDataSet || (indexInPidEntryArray - 1) < 0)
                    {
                        //we will use the true value for Y
                        var previousY = previousEntry.Y;
                        if (double.IsNaN(previousY))
                        {
                            throw new Exception("no Y value associated with entry " + (indexInPidEntryArray - 1) + " of pid " + pid);
                        }
                        xDest[idx++] = previousY;
                    }
                    else
                    {
                        //we need to use the estimated value for Y (even if the true value of Y is available)
                        var previousElementId = _CFM60EntryIDToElementId[previousEntry.ID];
                        if (!_elementIdToPrediction.ContainsKey(previousElementId))
                        {
                            throw new Exception("missing prediction for ID " + previousEntry.ID + " with pid " + pid + " : it is required to make the prediction for next ID " + entry.ID);
                        }
                        xDest[idx++] = _elementIdToPrediction[previousElementId];
                    }
                }

                if (Cfm60NetworkBuilder.Use_ret_vol_in_InputTensor)
                {
                    foreach (var ret_vol in entry.ret_vol)
                    {
                        xDest[idx++] = ret_vol;
                    }
                }
                if (Cfm60NetworkBuilder.Use_abs_ret_in_InputTensor)
                {
                    foreach (var abs_ret in entry.abs_ret)
                    {
                        xDest[idx++] = abs_ret;
                    }
                }

                if (Cfm60NetworkBuilder.Use_y_LinearRegressionEstimate_in_InputTensor)
                {
                    xDest[idx++] = CFM60TrainingAndTestDataSet.LinearRegressionEstimateBasedOnFullTrainingSet(entry.pid, entry.day);
                }
                if (Cfm60NetworkBuilder.Use_pid_y_avg_in_InputTensor)
                {
                    xDest[idx++] = CFM60TrainingAndTestDataSet.Y_Average_BasedOnFullTrainingSet(entry.pid);
                }
                if (Cfm60NetworkBuilder.Use_pid_y_vol_in_InputTensor)
                {
                    xDest[idx++] = CFM60TrainingAndTestDataSet.Y_Volatility_BasedOnFullTrainingSet(entry.pid);
                }
                if (Cfm60NetworkBuilder.Use_pid_y_variance_in_InputTensor)
                {
                    xDest[idx++] = CFM60TrainingAndTestDataSet.Y_Variance_BasedOnFullTrainingSet(entry.pid);
                }
                if (Cfm60NetworkBuilder.Use_ret_vol_CoefficientOfVariation_in_InputTensor)
                {
                    xDest[idx++] = entry.Get_ret_vol_CoefficientOfVariation();
                }
                if (Cfm60NetworkBuilder.Use_ret_vol_Volatility_in_InputTensor)
                {
                    xDest[idx++] = entry.Get_ret_vol_Volatility();
                }
                if (Cfm60NetworkBuilder.Use_LS_in_InputTensor)
                {
                    var ls = entry.LS;
                    if (Cfm60NetworkBuilder.NormalizeLS)
                    {
                        ls = NormalizeBetween_0_and_1(ls, ls_min, ls_max);
                    }

                    if (Cfm60NetworkBuilder.NormalizeLS_V2)
                    {
                        ls = (ls + 3.185075f) / 1.072115f;
                    }

                    xDest[idx++] = ls;
                }

                if (Cfm60NetworkBuilder.Use_NLV_in_InputTensor)
                {
                    var nlv = entry.NLV;
                    if (Cfm60NetworkBuilder.NormalizeNLV)
                    {
                        nlv = NormalizeBetween_0_and_1(nlv, nlv_min, nlv_max);
                    }

                    if (Cfm60NetworkBuilder.NormalizeNLV_V2)
                    {
                        nlv = (nlv + 0.018128f) / 1.002737f;
                    }

                    xDest[idx++] = nlv;
                }

                if (Cfm60NetworkBuilder.Use_day_in_InputTensor)
                {
                    xDest[idx++] = entry.day / Cfm60NetworkBuilder.Use_day_in_InputTensor_Divider;
                }

                if (Cfm60NetworkBuilder.Use_fraction_of_year_in_InputTensor)
                {
                    xDest[idx++] = DayToFractionOfYear(entry.day);
                }
                

                if (Cfm60NetworkBuilder.Use_EndOfYear_flag_in_InputTensor)
                {
                    xDest[idx++] = EndOfYear.Contains(entry.day) ? 1 : 0;
                }
                if (Cfm60NetworkBuilder.Use_Christmas_flag_in_InputTensor)
                {
                    xDest[idx++] = Christmas.Contains(entry.day) ? 1 : 0;
                }
                if (Cfm60NetworkBuilder.Use_EndOfTrimester_flag_in_InputTensor)
                {
                    xDest[idx++] = EndOfTrimester.Contains(entry.day) ? 1 : 0;
                }

                if (timeStep == 0 && idx != Cfm60NetworkBuilder.InputSize)
                {
                    throw new Exception("expecting "+ Cfm60NetworkBuilder.InputSize+" elements but got "+idx);
                }
            }

            if (yBuffer != null)
            {
                Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            var countInTraining = (int) (percentageInTrainingSet * Count);
            if (Cfm60NetworkBuilder.SplitTrainingAndValidationBasedOnDays)
            {
                var sortedDays = Entries.Select(e => e.day).OrderBy(x => x).ToArray();
                //dayThreshold = 651 => 80% in training, 20% in validation
                //dayThreshold = 690 => 85% in training, 15% in validation
                //dayThreshold = 728 => 90% in training, 10% in validation
                //dayThreshold = 766 => 95% in training,  5% in validation
                var dayThreshold = sortedDays[countInTraining];
                var training = new CFM60DataSet(Entries.Where(e => e.day <= dayThreshold).ToArray(), Cfm60NetworkBuilder);
                var validation = new CFM60DataSet(Entries.Where(e => e.day > dayThreshold).ToArray(), Cfm60NetworkBuilder, training);
                return new TrainingAndTestDataLoader(training, validation, Name);
            }
            else
            {
                var training = new CFM60DataSet(Entries.Take(countInTraining).ToArray(), Cfm60NetworkBuilder);
                var validation = new CFM60DataSet(Entries.Skip(countInTraining).ToArray(), Cfm60NetworkBuilder);
                Debug.Assert(Count == training.Count + validation.Count);
                return new TrainingAndTestDataLoader(training, validation, Name);
            }

        }

        public override int Count => Y.Shape[0];

        public override int ElementIdToCategoryIndex(int elementId)
        {
            return -1;
        }

        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public override CpuTensor<float> Y { get; }

        public override string ToString()
        {
            return Entries + " => " + Y;
        }

        public float ElementIdToExpectedAverage(int elementId)
        {
            var entryToPredict = _elementIdToEntryToPredict[elementId];
            var pid = entryToPredict.pid;
            var day = entryToPredict.day;
            return CFM60TrainingAndTestDataSet.LinearRegressionEstimateBasedOnFullTrainingSet(pid, day);
        }

        /// <summary>
        /// return the Mean Squared Error associated with the prediction 'idToPredictions'
        /// return value will be double.NaN if the MSE can not be computed
        /// </summary>
        /// <param name="CFM60EntryIDToPrediction">a prediction</param>
        /// <param name="applyLogToPrediction">true if we should apply log to the prediction before computing the MSE</param>
        /// <returns>
        /// The Mean Squared Error of the predictions, or double.NaN if was not computed
        /// </returns>
        public double ComputeMeanSquareError(IDictionary<int, double> CFM60EntryIDToPrediction, bool applyLogToPrediction)
        {
            var CFM60EntryIDToCFM60Entry = new Dictionary<int, CFM60Entry>();
            foreach(var entry in Entries)
            {
                CFM60EntryIDToCFM60Entry[entry.ID] = entry;
            }
            double mse = 0; //the mean squared error
            foreach (var (id, value) in CFM60EntryIDToPrediction)
            {
                var predictedValue = applyLogToPrediction ? Math.Log(value) : value;
                var expectedValue = CFM60EntryIDToCFM60Entry[id].Y;
                if (double.IsNaN(expectedValue))
                {
                    Log.Error("no known expected value for id:" + id);
                    return double.NaN;
                }
                mse += Math.Pow(predictedValue - expectedValue, 2);
            }
            return mse / CFM60EntryIDToPrediction.Count;
        }

        public static IDictionary<int, double> LoadPredictionFile(string filePath)
        {
            var predictions = new Dictionary<int, double>();
            if (string.IsNullOrEmpty(filePath))
            {
                return predictions;
            }
            foreach (var l in File.ReadAllLines(filePath).Skip(1))
            {
                var splitted = l.Split(',');
                predictions[int.Parse(splitted[0])] = double.Parse(splitted[1]);
            }
            return predictions;
        }
        // ReSharper disable once UnusedMember.Global
        public void ComputePredictions(Func<CFM60Entry, double> entryToPrediction, string comment)
        {
            var CFM60EntryIDToPrediction = new ConcurrentDictionary<int, double>();
            void ComputePrediction(int elementId)
            {
                var entryToPredict = _elementIdToEntryToPredict[elementId];
                var prediction = entryToPrediction(entryToPredict);
                CFM60EntryIDToPrediction[entryToPredict.ID] = prediction;
            }
            System.Threading.Tasks.Parallel.For(0, Count, ComputePrediction);
            CreatePredictionFile(CFM60EntryIDToPrediction, Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60", "PerformPrediction", comment + "_" + DateTime.Now.Ticks + ".csv"));
            //we update the file with all predictions
            var mse = ComputeMeanSquareError(CFM60EntryIDToPrediction, false);
            var testsCsv = Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60", "PerformPrediction", "Tests_CFM60_PerformPrediction.csv");
            var line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";" + comment.Replace(';', '_') + ";" + mse.ToString(CultureInfo.InvariantCulture) + ";" + Environment.NewLine;
            File.AppendAllText(testsCsv, line);
        }

        private static float NormalizeBetween_0_and_1(float initialValue, float knownMinValue, float knownMaxValue)
        {
            if (knownMinValue >= knownMaxValue)
            {
                return 0; //constant
            }

            return (initialValue - knownMinValue) / (knownMaxValue - knownMinValue);
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
                var pid = _elementIdToEntryToPredict[elementId].pid;
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
