using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Networks;
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

namespace SharpNet.Datasets
{

    public class CFM60Entry
    {
        public const int POINTS_BY_DAY = 61;

        public CFM60Entry(string line)
        {
            var splitted = line.Split(',');
            int index = 0;
            ID = int.Parse(splitted[index++]);
            pid = int.Parse(splitted[index++]);
            day = int.Parse(splitted[index++]);
            abs_ret = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    abs_ret[i] = (float)tmp;
                }
            }
            ret_vol = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    ret_vol[i] = (float)tmp;
                }
            }
            LS = double.Parse(splitted[index++]);
            NLV = double.Parse(splitted[index]);
        }

        public double ret_vol_from_last(double last)
        {
            int entireCount = (int) (last + 0.00001);
            double result = 0;
            for (int i = 0; i < entireCount; ++i)
            {
                result += ret_vol[ret_vol.Length - 1 - i];
            }
            result += (last-result)*ret_vol[ret_vol.Length - 1 - entireCount];
            return result;
        }

        public double last_ret_vol_multiplied(double multiplier)
        {
            return multiplier*ret_vol.Last();
        }

        public int ID { get;  }
        public int pid { get; }
        public int day { get; }
        public float[] abs_ret { get; }
        public float[] ret_vol{ get; }
        public double LS { get; }
        public double NLV { get; }

    }

    public class CFM60DataSet : AbstractDataSet
    {
        #region private fields
        [NotNull]
        private readonly  CFM60Entry[] _entries;
        private readonly IDictionary<int, double> _y;
        #endregion


        public void CreateSummaryFile(string filePath)
        {
            var sb = new StringBuilder();
            sb.Append("sep=,");
            sb.Append(Environment.NewLine + "ID,pid,day,expected");
            foreach (var e in Entries)
            {
                sb.Append(Environment.NewLine + e.ID + "," + e.pid + "," + e.day);
                if (_y != null && _y.TryGetValue(e.ID, out var expected))
                {
                    sb.Append("," + expected.ToString(CultureInfo.InvariantCulture));
                }
            }
            File.WriteAllText(filePath, sb.ToString());
        }

        public CFM60Entry[] Entries => _entries;
        //public IDictionary<int, double> Expected => _y;

        public double ComputeMeanSquareError(IDictionary<int, double> predictions, bool applyLogToPrediction)
        {
            double error = 0;
            if (predictions.Count != _y.Count)
            {
                throw new ArgumentException("invalid predictions of length "+predictions.Count);
            }
            foreach (var p in predictions)
            {
                var predictedValue = p.Value;
                if (applyLogToPrediction)
                {
                    predictedValue = Math.Log(predictedValue);
                }
                var expectedValue = _y[p.Key];
                error += Math.Pow(predictedValue - expectedValue, 2);
            }
            return error / predictions.Count;
        }

        public static void CreatePredictionFile(IDictionary<int, double> predictions, string filePath)
        {
            var sb = new StringBuilder();
            sb.Append("ID,target");
            foreach (var p in predictions.OrderBy(x => x.Key))
            {
                sb.Append(Environment.NewLine+p.Key+","+p.Value.ToString(CultureInfo.InvariantCulture));
            }
            File.WriteAllText(filePath, sb.ToString());
        }

        public static IDictionary<int, double> LoadPredictionFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                return null;
            }

            Log.Info("Loading prediction file " + filePath + "...");
            var predictions = new Dictionary<int, double>();
            foreach (var l in File.ReadAllLines(filePath).Skip(1))
            {
                var splitted = l.Split(',');
                predictions[int.Parse(splitted[0])] = double.Parse(splitted[1]);
            }

            Log.Info("Prediction file " + filePath + " has been loaded ("+predictions.Count+" entries)");
            return predictions;
        }

        public CFM60DataSet(string xFile, string yFileIfAny)
            : base("CFM60",
                CFM60Entry.POINTS_BY_DAY,
                new []{"NONE"},
                null,
                ResizeStrategyEnum.None)
        {
            Debug.Assert(xFile != null);

            Log.Info("Loading content of file "+ xFile+"...");
            var xFileContent = File.ReadAllLines(xFile);

            Log.Info("File " + xFile + " has been loaded: "+xFileContent.Length+" lines");
            _entries = new CFM60Entry[xFileContent.Length-1];

            Log.Info("Parsing lines of file " + xFile + "...");
            void ProcessLine(int i)
            {
                Debug.Assert(i>=1);
                _entries[i-1] = new CFM60Entry(xFileContent[i]);
            }
            Log.Info("Lines of file " + xFile + " have been parsed");

            System.Threading.Tasks.Parallel.For(1, xFileContent.Length, ProcessLine);

            _y = LoadPredictionFile(yFileIfAny);

            if (_y != null && _y.Count != 0)
            {
                var expected = new float[Count];
                for (int i = 0; i < _entries.Length; ++i)
                {
                    expected[i] = (float)_y[_entries[i].ID];
                }
                Y= new CpuTensor<float>(new []{Count, 1}, expected);
            }
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            //Debug.Assert(xBuffer.SameShapeExceptFirstDimension(X_Shape));
            Debug.Assert(xBuffer.Shape[0] == yBuffer.Shape[0]); //same batch size
            Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));

            var xSrc = new Span<float>(_entries[elementId].ret_vol);
            var xDest = xBuffer.AsFloatCpuSpan.Slice(indexInBuffer * xBuffer.MultDim0, xBuffer.MultDim0);
            Debug.Assert(xSrc.Length == xDest.Length);
            xSrc.CopyTo(xDest);
            if (yBuffer != null)
            {
                Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        public override int Count => _entries.Length;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return -1;
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }


        public int TimeSteps => 61;
        public int InputSize => 1;

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _entries + " => " + Y;
        }

        public void PerformPrediction(Func<CFM60Entry, double> entryToPrediction, string comment)
        {
            Log.Info("performing prediction "+comment);
            var predictions = new ConcurrentDictionary<int, double>();
            void ComputePrediction(int i)
            {
                var prediction = entryToPrediction(Entries[i]);
                predictions[Entries[i].ID] = prediction;
            }

            System.Threading.Tasks.Parallel.For(0, Entries.Length, ComputePrediction);

            if (_y != null)
            {
                var mse = ComputeMeanSquareError(predictions, false);
                Log.Info("Prediction MSE " + mse);
                var testsCsv = Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60", "Tests_CFM60.csv");
                try
                {
                    //We save the results of the net
                    var line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                                                                                           + comment.Replace(';', '_') + ";"
                                                                                           + "DeviceName" + ";"
                                                                                           + "TotalParams" + ";"
                                                                                           + "numEpochs" + ";"
                                                                                           + "miniBatchSizeForAllWorkers" + ";"
                                                                                           + "learningRate" + ";"
                                                                                           + "0;"
                                                                                           + "0;"
                                                                                           + mse.ToString(CultureInfo.InvariantCulture) + ";"
                                                                                           + Environment.NewLine;
                    File.AppendAllText(testsCsv, line);
                }
                catch (Exception e) 
                {
                    Log.Error("fail to write to file "+testsCsv+Environment.NewLine+e);
                }
            }


            Log.Info("creating  prediction file for " + comment);
            CreatePredictionFile(predictions, "c:/temp/"+comment+"_"+DateTime.Now.Ticks+".csv");
        }
    }


    public class CFM60TrainingAndTestDataSet : AbstractTrainingAndTestDataSet
    {
        // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
        private readonly bool _useFullDataSet;
        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
        public override int CategoryByteToCategoryIndex(byte categoryByte) {return -1;}
        public override byte CategoryIndexToCategoryByte(int categoryIndex) {return 0;}
        public CFM60TrainingAndTestDataSet(bool useFullDataSet) : base("CFM60")
        {
            _useFullDataSet = useFullDataSet;

            if (_useFullDataSet)
            {
                Training = new CFM60DataSet(
                    Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "input_training.csv"),
                    Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "output_training_IxKGwDV.csv"));
                Test = new CFM60DataSet(
                    Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "input_test.csv"),
                    null);
            }
            else
            {
                //we'll use a DataSet with 100K entries
                Training = new CFM60DataSet(
                    Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "input_training_small.csv"),
                    Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "output_training_small.csv"));
                // ReSharper disable once VirtualMemberCallInConstructor
                Test = Training;
            }
        }
    }
}
