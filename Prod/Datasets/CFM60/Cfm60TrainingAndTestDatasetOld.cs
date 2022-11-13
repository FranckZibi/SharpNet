using System;
using System.IO;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM60
{
    public class Cfm60TrainingAndTestDatasetOld : AbstractTrainingAndTestDataset
    {
        public override DataSet Training { get; }
        public override DataSet Test { get; }

        public override int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return -1;
        }

        public override byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return 0;
        }

        public Cfm60TrainingAndTestDatasetOld(Cfm60NetworkSampleOld cfm60NetworkSampleOld, Action<string> log) : base("CFM60")
        {
            Training = new CFM60DataSetOld(
                Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_training.csv"),
                Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "output_training_IxKGwDV.csv"),
                log,
                cfm60NetworkSampleOld,
                null);
            Test = new CFM60DataSetOld(Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_test.csv"),
                null,
                log,
                cfm60NetworkSampleOld,
                // ReSharper disable once VirtualMemberCallInConstructor
                (CFM60DataSetOld)Training);
            //CFM60Utils.Create_Summary_File(((CFM60DataSet)Training).Entries.Union(((CFM60DataSet)Test).Entries).ToList());
        }
    }
}