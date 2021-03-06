﻿namespace SharpNet.Datasets
{
    public class ZoomedTrainingAndTestDataSet : AbstractTrainingAndTestDataSet
    {
        private readonly AbstractTrainingAndTestDataSet _original;

        public ZoomedTrainingAndTestDataSet(AbstractTrainingAndTestDataSet original, int[] originalShape_CHW, int heightMultiplier, int widthMultiplier) 
            : base(original.Name)
        {
            _original = original;
            Training = new ZoomedDataSet(original.Training, originalShape_CHW, heightMultiplier, widthMultiplier);
            Test = new ZoomedDataSet(original.Test, originalShape_CHW, heightMultiplier, widthMultiplier);
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
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