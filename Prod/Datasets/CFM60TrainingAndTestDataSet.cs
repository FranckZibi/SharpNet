using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public class CFM60TrainingAndTestDataSet : AbstractTrainingAndTestDataSet
    {
        private static IDictionary<int, LinearRegression> PidToLinearRegressionBetweenDayAndY;
        public override IDataSet Training { get; }
        public override IDataSet Test { get; }

        public override int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return -1;
        }

        public override byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return 0;
        }

        public CFM60TrainingAndTestDataSet(CFM60NetworkBuilder cfm60NetworkBuilder, Action<string> log) : base("CFM60")
        {
            Training = new CFM60DataSet(
                Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "input_training.csv"),
                Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "output_training_IxKGwDV.csv"),
                log,
                cfm60NetworkBuilder,
                null);
            Test = new CFM60DataSet(Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "input_test.csv"),
                null, 
                log,
                cfm60NetworkBuilder,
                (CFM60DataSet)Training);

            lock (lockObject)
            {
                if (PidToLinearRegressionBetweenDayAndY == null)
                {
                    // ReSharper disable once VirtualMemberCallInConstructor
                    PidToLinearRegressionBetweenDayAndY = ((CFM60DataSet) Training).ComputePidToLinearRegressionBetweenDayAndY();
                }
            }
        }

        public static float LinearRegressionEstimateBasedOnFullTrainingSet(int pid, int day)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Estimation(day);
        }
        public static float Y_Mean_BasedOnFullTrainingSet(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Average;
        }
        public static float Y_Volatility_BasedOnFullTrainingSet(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Volatility;
        }
        public static float Y_Variance_BasedOnFullTrainingSet(int pid)
        {
            return (float)PidToLinearRegressionBetweenDayAndY[pid].Y_Variance;
        }

        private static readonly object lockObject = new object();
    }
}