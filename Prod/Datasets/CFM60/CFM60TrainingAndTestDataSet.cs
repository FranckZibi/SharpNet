using System;
using System.IO;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM60
{
    public class CFM60TrainingAndTestDataSet : AbstractTrainingAndTestDataSet
    {
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
                // ReSharper disable once VirtualMemberCallInConstructor
                (CFM60DataSet) Training);
            //CFM60Utils.Create_Summary_File(((CFM60DataSet)Training).Entries.Union(((CFM60DataSet)Test).Entries).ToList());
        }
    }
}