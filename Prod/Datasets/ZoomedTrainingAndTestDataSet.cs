namespace SharpNet.Datasets
{
    public class ZoomedTrainingAndTestDataset : AbstractTrainingAndTestDataset
    {
        private readonly AbstractTrainingAndTestDataset _original;

        public ZoomedTrainingAndTestDataset(AbstractTrainingAndTestDataset original, int[] originalShape_CHW, int heightMultiplier, int widthMultiplier) 
            : base(original.Name)
        {
            _original = original;
            Training = new ZoomedDataSet(original.Training, originalShape_CHW, heightMultiplier, widthMultiplier);
            Test = new ZoomedDataSet(original.Test, originalShape_CHW, heightMultiplier, widthMultiplier);
        }

        public override DataSet Training { get; }
        public override DataSet Test { get; }
        public override int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return _original.CategoryByteToCategoryIndex(categoryByte);
        }
        public override byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return _original.CategoryIndexToCategoryByte(categoryIndex);
        }
    }
}