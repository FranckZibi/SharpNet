using System.Collections.Generic;
using System;

namespace SharpNet.Datasets
{
    public abstract class AbstractTrainingAndTestDataset : ITrainingAndTestDataset
    {
        #region public properties
        public abstract DataSet Training { get; }
        public abstract DataSet Test { get; }
        public string Name { get; }
        #endregion

        #region constructor
        protected AbstractTrainingAndTestDataset(string name)
        {
            Name = name;
        }
        #endregion

        public virtual void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }
        public virtual int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return categoryByte;
        }
        public virtual byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return (byte)categoryIndex;
        }



        public AbstractTrainingAndTestDataset WithRandomizeColumnDataSet(List<string> columnsToRandomize, Random r)
        {
            if (columnsToRandomize.Count == 0)
            {
                return this;
            }
            var randomizedTraining = Training == null ? null : new RandomizeColumnDataSet(Training, columnsToRandomize, r);
            var randomizedValidation = Test == null ? null : new RandomizeColumnDataSet(Test, columnsToRandomize, r);
            return new TrainingAndTestDataset(randomizedTraining, randomizedValidation, nameof(RandomizeColumnDataSet)+"_");
        }


    }
}
