namespace SharpNet.Datasets
{
    public class TrainingAndTestDataset : AbstractTrainingAndTestDataset
    {
        public TrainingAndTestDataset(DataSet training, DataSet test, string name)
            : base(name)
        {
            Training = training;
            Test = test;
        }

        public override DataSet Training { get; }
        public override DataSet Test { get; }
    }
}