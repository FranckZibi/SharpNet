using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.CFM60;

public class CFM60DataSetOld : DataSet, ITimeSeriesDataSet
{
    private readonly CpuTensor<float> _yCFM60DataSetOld;

    public Cfm60NetworkSampleOld SampleOld => _cfm60NetworkSampleOld;
    private readonly Cfm60NetworkSampleOld _cfm60NetworkSampleOld;
    private readonly CFM60DataSetOld _trainingDataSetOldIfAny;
    //TODO : add element for each valid day, even if no entry is associated with that day
    private readonly Dictionary<int, List<CFM60Entry>> _pidToSortedEntries = new();
    private readonly Dictionary<int, int> _CFM60EntryIDToIndexIn_pidToSortedEntries = new();
    private readonly Dictionary<int, CFM60Entry> _elementIdToLastAssociateCFM60Entry = new();

    // CFM60EntryID = CFM60Entry.ID: the unique ID of a CFM60Entry
    // elementId : id of an element in the dataSet (in range [0, dataSet.Count[ )
    private readonly IDictionary<int, float> _idToPrediction = new Dictionary<int, float>();

    
    #region private fields

    [NotNull] public readonly CFM60Entry[] Entries;


    ///// <summary>
    ///// Statistics of the expected outcome 'y' of the all dataSet (E(y), Vol(y), Max, Min)
    ///// </summary>
    //private readonly DoubleAccumulator Y_stats = new DoubleAccumulator();

    #endregion

    public CFM60DataSetOld(string xFile, string yFileIfAny, Action<string> log, Cfm60NetworkSampleOld sampleOld, CFM60DataSetOld trainingDataSetOldIfAny = null)
        : this(CFM60Entry.Load(xFile, yFileIfAny, log), sampleOld, trainingDataSetOldIfAny)
    {
    }

    public CFM60DataSetOld(CFM60Entry[] entries, Cfm60NetworkSampleOld cfm60NetworkSampleOld, CFM60DataSetOld trainingDataSetOldIfAny = null)
        : base("CFM60",
            Objective_enum.Regression,
            cfm60NetworkSampleOld.Encoder_TimeSteps,
            null,
            ResizeStrategyEnum.None,
            cfm60NetworkSampleOld.ComputeFeatureNames(),
            Array.Empty<string>(),
            "",
            null, //TODO
            UseBackgroundThreadToLoadNextMiniBatch(trainingDataSetOldIfAny),
            ',')
    {
        _cfm60NetworkSampleOld = cfm60NetworkSampleOld;
        _trainingDataSetOldIfAny = trainingDataSetOldIfAny;
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

        //we initialize _IDToIndexIn_pidToSortedEntries
        foreach (var e in _pidToSortedEntries.Values)
        {
            for (int index_in_pidToSortedEntries = 0; index_in_pidToSortedEntries < e.Count; ++index_in_pidToSortedEntries)
            {
                _CFM60EntryIDToIndexIn_pidToSortedEntries[e[index_in_pidToSortedEntries].ID] = index_in_pidToSortedEntries;
            }
        }

        //we initialize _elementIdToAssociateLastCFM60Entry
        int longestEntry = _pidToSortedEntries.Values.Select(x => x.Count).Max();
        int[] pids = _pidToSortedEntries.Keys.OrderBy(x => x).ToArray();
        int idxLastEntry = IsTrainingDataSet
            ? EntriesCountForEachElementId_X + EntriesCountForEachElementId_Y - 1
            : EntriesCountForEachElementId_Y - 1;
        while (idxLastEntry < longestEntry)
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
            foreach (var e in ElementId_to_YEntries(elementId))
            {
                yData[nextIdxInY++] = e.Y;
            }
        }
        Debug.Assert(nextIdxInY == yData.Length);
        _yCFM60DataSetOld = new CpuTensor<float>(new[] { count, EntriesCountForEachElementId_Y }, yData);

        //if we are in a training data set
        if (trainingDataSetOldIfAny == null)
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
        }
        else //validation or test data set
        {
            //we ensure that the associate training data set is valid
            foreach (var (pid, validationEntries) in _pidToSortedEntries)
            {
                if (!trainingDataSetOldIfAny._pidToSortedEntries.ContainsKey(pid))
                {
                    throw new Exception("validation pid " + pid + " doesn't exist in training data set");
                }
                if (validationEntries.Count == 0)
                {
                    throw new Exception("not enough entries (" + validationEntries.Count + ") in validation data set for pid " + pid);
                }
            }
        }

    }

    //public int Encoder_TimeSteps => Cfm60NetworkBuilder.Encoder_TimeSteps;
    public int Total_TimeSteps => SampleOld.Total_TimeSteps;

    int EntriesCountForEachElementId_X => SampleOld.Use_Decoder ? SampleOld.Encoder_TimeSteps : 1 + SampleOld.Encoder_TimeSteps;

    int EntriesCountForEachElementId_Y => SampleOld.Use_Decoder ? SampleOld.Decoder_TimeSteps : 1;

    public IEnumerable<CFM60Entry> ElementId_to_YEntries(int elementId)
    {
        var lastEntry = _elementIdToLastAssociateCFM60Entry[elementId];
        var pidEntries = _pidToSortedEntries[lastEntry.pid];
        var lastIdx = _CFM60EntryIDToIndexIn_pidToSortedEntries[lastEntry.ID];
        int firstIdx = lastIdx - EntriesCountForEachElementId_Y + 1;
        for (int idx = firstIdx; idx <= lastIdx; ++idx)
        {
            yield return pidEntries[idx];
        }
    }

    public void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions)
    {
        Debug.Assert(batchPredictions.Count == batchElementIds.Length * EntriesCountForEachElementId_Y);
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
    public CFM60DataSetOld ValidationDataSetOld { get; set; }

    /// <summary>
    /// the original (and complete) Test Data Set for the CFM60 challenge
    /// </summary>
    public CFM60DataSetOld OriginalTestDataSetOld { get; set; }

    /// <summary>
    /// for Neural network model:
    ///     will save also the pid features, and the prediction file for the Train + Validation + Test datasets
    /// </summary>
    public override void Save(Model model, string workingDirectory, string modelName)
    {
        base.Save(model, workingDirectory, modelName);
        CreatePredictionFile(model, "_predict_train_");
        ValidationDataSetOld?.CreatePredictionFile(model, "_predict_valid_");
        OriginalTestDataSetOld?.CreatePredictionFile(model, "_predict_test_");

        if (model is Network network)
        {

            var embeddingLayer = network.Layers.FirstOrDefault(l => l is EmbeddingLayer);
            if (embeddingLayer == null)
            {
                return;
            }
            var cpuTensor = embeddingLayer.Weights.ToCpuFloat();
            cpuTensor.Save(Path.Combine(network.WorkingDirectory, "pid_features_" + network.ModelName + ".csv"),
                row => row >= 1 && row <= CFM60Entry.DISTINCT_PID_COUNT, //the first row is not used in word embedding
                true,
                "pid;" + string.Join(";", Enumerable.Range(0, cpuTensor.Shape[0]).Select(i => "feature_" + i))
            );
            return;
        }
        throw new ArgumentException($"cant' save model of type {model.GetType()}");
    }

    public void CreatePredictionFile(Model model, string fileSuffix)
    {
        var res = model.Predict(this, false);
        var CFM60EntryIDToPrediction = new Dictionary<int, double>();
        var spanResult = res.FloatCpuTensor().ReadonlyContent;
        for (int elementId = 0; elementId < Count; ++elementId)
        {
            var id = _elementIdToLastAssociateCFM60Entry[elementId].ID;
            var prediction = spanResult[elementId];
            CFM60EntryIDToPrediction[id] = prediction;
        }
        string filePath = Path.Combine(model.WorkingDirectory, model.ModelName + fileSuffix + ".csv");
        SavePredictions(CFM60EntryIDToPrediction, filePath);
    }

    #region private fields
    private static readonly Dictionary<int, CFM60Entry> id_to_entries = Load_Summary_File();
    #endregion

    private static Dictionary<int, CFM60Entry> Load_Summary_File()
    {
        var res = new Dictionary<int, CFM60Entry>();
        var path = Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "CFM60_summary.csv");
        foreach (var l in File.ReadAllLines(path).Skip(1))
        {
            var entry = new CFM60Entry();
            var splitted = l.Split(',', ';');
            entry.ID = int.Parse(splitted[0]);
            entry.pid = int.Parse(splitted[1]);
            entry.day = int.Parse(splitted[2]);
            entry.Y = entry.LS = entry.NLV = float.NaN;
            if (float.TryParse(splitted[3], out var tmp))
            {
                entry.Y = tmp;
            }
            if (splitted.Length >= 5)
            {
                entry.LS = float.Parse(splitted[4]); //we store in the LS field the linear regression of Y
            }
            if (splitted.Length >= 6)
            {
                entry.NLV = float.Parse(splitted[5]); //we store in the NLV field the adjusted linear regression of Y
            }
            res[entry.ID] = entry;
        }
        return res;
    }


    public static void SavePredictions(IDictionary<int, double> CFM60EntryIDToPrediction, string filePath, double multiplierCorrection = 1.0, double addCorrectionStart = 0.0, double addCorrectionEnd = 0.0)
    {
        var sb = new StringBuilder();
        sb.Append("ID,target");
        foreach (var (id, originalPrediction) in CFM60EntryIDToPrediction.OrderBy(x => x.Key))
        {
            if (CFM60Entry.IsInterpolatedId(id))
            {
                continue;
            }
            var day = id_to_entries[id].day;
            var fraction = (day - 805.0) / (1151 - 805.0);
            var toAdd = addCorrectionStart + (addCorrectionEnd - addCorrectionStart) * fraction;
            var prediction = multiplierCorrection * originalPrediction + toAdd;
            sb.Append(Environment.NewLine + id + "," + prediction.ToString(CultureInfo.InvariantCulture));
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
        var trainingEntries = _trainingDataSetOldIfAny._pidToSortedEntries[pid];
        return trainingEntries[trainingEntries.Count + indexInPidEntryArray];
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
        bool withDataAugmentation, bool isTraining)
    {
        throw new Exception("should never be called");
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="elementId">gives the entry to predict</param>
    /// <param name="indexInBuffer"></param>
    /// <param name="all_xBuffer">
    ///     input shape: (batchSize, TimeSteps, InputSize)</param>
    /// <param name="yBuffer">
    ///     output shape: (batchSize, 1)
    /// </param>
    /// <param name="withDataAugmentation"></param>
    /// <param name="isTraining"></param>
    protected override void LoadAt(int elementId, int indexInBuffer, List<CpuTensor<float>> all_xBuffer,
        CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        var xEncoder = all_xBuffer[0];
        Debug.Assert(xEncoder.Shape.Length == 3);
        Debug.Assert(indexInBuffer >= 0 && indexInBuffer < all_xBuffer[0].Shape[0]);
        Debug.Assert(xEncoder.Shape[1] == SampleOld.Encoder_TimeSteps);
        Debug.Assert(xEncoder.Shape[2] == SampleOld.Encoder_InputSize);
        Debug.Assert(yBuffer == null || all_xBuffer[0].Shape[0] == yBuffer.Shape[0]); //same batch size
        Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(_yCFM60DataSetOld.Shape));
        Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(_yCFM60DataSetOld.Shape));

        LoadAt(elementId, indexInBuffer, xEncoder, true);
        if (SampleOld.Use_Decoder)
        {
            Debug.Assert(all_xBuffer.Count == 2);
            var xDecoder = all_xBuffer[1];
            LoadAt(elementId, indexInBuffer, xDecoder, false);
        }

        if (yBuffer != null)
        {
            _yCFM60DataSetOld.CopyTo(_yCFM60DataSetOld.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
        }
    }

    private void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> x, bool isEncoder)
    {
        var xElementId = x.AsFloatCpuSpan.Slice(indexInBuffer * x.MultDim0, x.MultDim0);

        int idx = 0;
        CFM60Entry lastAssociateCFM60Entry = _elementIdToLastAssociateCFM60Entry[elementId];
        var pid = lastAssociateCFM60Entry.pid;
        int lastIndexInPidEntries = _CFM60EntryIDToIndexIn_pidToSortedEntries[lastAssociateCFM60Entry.ID];
        Debug.Assert(_pidToSortedEntries[pid][lastIndexInPidEntries].ID == lastAssociateCFM60Entry.ID);
        if (isEncoder && SampleOld.Use_Decoder)
        {
            Debug.Assert(SampleOld.Decoder_TimeSteps >= 1);
            lastIndexInPidEntries -= SampleOld.Decoder_TimeSteps;
        }

        int timeSteps = x.Shape[1];
        // ReSharper disable once LoopVariableIsNeverChangedInsideLoop
        for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
        {
            var indexInPidEntryArray = lastIndexInPidEntries - timeSteps + timeStep + 1;
            var entry = GetEntry(pid, indexInPidEntryArray);

            //pid
            if (SampleOld.Pid_EmbeddingDim >= 1)
            {
                //pids are in range  [0, 899]
                //EmbeddingLayer is expecting them in range [1,900] that's why we add +1
                xElementId[idx++] = entry.pid + 1;
            }

            //day/year
            if (SampleOld.Use_day)
            {
                xElementId[idx++] = entry.day / SampleOld.Use_day_Divider;
            }
            if (SampleOld.Use_fraction_of_year)
            {
                xElementId[idx++] = CFM60DatasetSample.DayToFractionOfYear(entry.day);
            }
            if (SampleOld.Use_year_Cyclical_Encoding)
            {
                xElementId[idx++] = (float)Math.Sin(2 * Math.PI * CFM60DatasetSample.DayToFractionOfYear(entry.day));
                xElementId[idx++] = (float)Math.Cos(2 * Math.PI * CFM60DatasetSample.DayToFractionOfYear(entry.day));
            }
            if (SampleOld.Use_EndOfYear_flag)
            {
                xElementId[idx++] = CFM60DatasetSample.EndOfYear.Contains(entry.day) ? 1 : 0;
            }
            if (SampleOld.Use_Christmas_flag)
            {
                xElementId[idx++] = CFM60DatasetSample.Christmas.Contains(entry.day) ? 1 : 0;
            }
            if (SampleOld.Use_EndOfTrimester_flag)
            {
                xElementId[idx++] = CFM60DatasetSample.EndOfTrimester.Contains(entry.day) ? 1 : 0;
            }
            //abs_ret
            if (SampleOld.Use_abs_ret)
            {
                //entry.abs_ret.AsSpan().CopyTo(xDest.Slice(idx, entry.abs_ret.Length));
                //idx += entry.abs_ret.Length;
                for (int i = 0; i < entry.abs_ret.Length; ++i)
                {
                    xElementId[idx++] = entry.abs_ret[i];
                }
            }
            //rel_vol
            if (SampleOld.Use_rel_vol)
            {
           
                //asSpan.CopyTo(xDest.Slice(idx, entry.rel_vol.Length));
                //idx += entry.rel_vol.Length;
                for (int i = 0; i < entry.rel_vol.Length; ++i)
                {
                    xElementId[idx++] = entry.rel_vol[i];
                }
            }
            //LS
            if (SampleOld.Use_LS)
            {
                xElementId[idx++] = entry.LS;
            }
            //NLV
            if (SampleOld.Use_NLV)
            {
                xElementId[idx++] = entry.NLV;
            }
            //y estimate
            if (SampleOld.Use_prev_Y && isEncoder)
            {
                var indexOfyEntryInPidEntryArray = SampleOld.Use_Decoder
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
                    xElementId[idx++] = y;
                }
                else
                {
                    //we need to use the estimated value for Y (even if the true value of Y is available)
                    if (!_idToPrediction.ContainsKey(yEntry.ID))
                    {
                        throw new Exception("missing prediction for ID " + yEntry.ID + " with pid " + pid + " : it is required to make the prediction for next ID " + entry.ID);
                    }
                    xElementId[idx++] = _idToPrediction[yEntry.ID];
                }
            }

            int expectedInputSize = isEncoder ? SampleOld.Encoder_InputSize : SampleOld.Decoder_InputSize;
            if (timeStep == 0 && elementId == 0 && idx != expectedInputSize)
            {
                throw new Exception("expecting " + expectedInputSize + " elements but got " + idx);
            }
        }
    }
    public override ITrainingAndTestDataset SplitIntoTrainingAndValidation(double percentageInTrainingSet)
    {
        var dayThreshold = CFM60DatasetSample.DayThreshold(Entries, percentageInTrainingSet);
        var training = new CFM60DataSetOld(Entries.Where(e => e.day <= dayThreshold).ToArray(), _cfm60NetworkSampleOld);
        var validation = new CFM60DataSetOld(Entries.Where(e => e.day > dayThreshold).ToArray(), _cfm60NetworkSampleOld, training);
        return new TrainingAndTestDataset(training, validation, Name);
    }
    public override DataSet SubDataSet(double percentageToKeep)
    {
        throw new NotImplementedException();
    }
    /// <summary>
    /// we'll save the network if we have reached a very small loss
    /// </summary>
    public override bool ShouldCreateSnapshotForEpoch(int epoch, Network network)
    {
        return epoch >= 2
               && network.CurrentEpochIsAbsolutelyBestInValidationLoss()
               && !double.IsNaN(network.EpochData.Last().GetValidationLoss(network.Sample.LossFunction))
               && network.Sample.AlwaysUseFullTestDataSetForLossAndAccuracy
               && network.EpochData.Last().GetValidationLoss(network.Sample.LossFunction) < SampleOld.MaxLossToSaveTheNetwork;
    }
    public override int Count => _yCFM60DataSetOld.Shape[0];
    public override int ElementIdToCategoryIndex(int elementId)
    {
        return -1;
    }
    public override double PercentageToUseForLossAndAccuracyFastEstimate => 0.0; //we do not compute any estimate
    public override CpuTensor<float> Y => _yCFM60DataSetOld;
    public override string ToString()
    {
        var xShape = new[] { Count, SampleOld.Encoder_TimeSteps, SampleOld.Encoder_InputSize};
        return Tensor.ShapeToString(xShape) + " => " + Tensor.ShapeToString(_yCFM60DataSetOld.Shape);
    }
    private static bool UseBackgroundThreadToLoadNextMiniBatch(CFM60DataSetOld trainingDataSetOldIfAny)
    {
        if (trainingDataSetOldIfAny != null)
        {
            //for Validation/Test DataSet, we should not use a background thread for loading next mini batch data
            return false;
        }
        //for Training DataSet, we should use background thread for loading next mini batch
        return true;
    }
    public bool IsTrainingDataSet => _trainingDataSetOldIfAny == null;
    public override List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer)
    {
        var result = new List<int[]> { shapeForFirstLayer };
        if (SampleOld.Use_Decoder)
        {
            var inputShapeDecoder = new[]
            {
                shapeForFirstLayer[0],
                SampleOld.Decoder_TimeSteps,
                SampleOld.Decoder_InputSize
            };
            result.Add(inputShapeDecoder);
        }
        return result;
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
