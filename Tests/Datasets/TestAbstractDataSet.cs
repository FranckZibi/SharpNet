using System;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNetTests.CPU;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestAbstractDataSet
    {
        [Test]
        public void TestOriginalElementContent()
        {
            var rand = new Random(0);
            var shape = new[] { 10, 3, 32, 32 };
            var tensorX = TestCpuTensor.RandomByteTensor(shape, rand, 0, 255);
            var dataSet = GetRandomDataSet(tensorX, 2, rand);
            for (int elementId = 0; elementId < shape[0]; ++elementId)
            {
                var bmp = dataSet.OriginalElementContent(elementId, shape[1], shape[2], shape[3], false, false);
                for (int channel = 0; channel < shape[1]; ++channel)
                {
                    for (int row = 0; row < shape[2]; ++row)
                    {
                        for (int col = 0; col < shape[3]; ++col)
                        {
                            Assert.AreEqual(tensorX.Get(elementId, channel, row, col), bmp.Get(channel, row, col));
                        }
                    }
                }
            }
        }

        private static DataSet GetRandomDataSet(CpuTensor<byte> tensorX, int numClass, Random rand)
        {
            var tensorY = TestCpuTensor.RandomByteTensor(new[] { tensorX.Shape[0], 1 }, rand, 0, 1);

            var meanAndVolatilityForEachChannel = tensorX.ComputeMeanAndVolatilityOfEachChannel(t => t);
            var x = DataSet.ToXWorkingSet(tensorX, meanAndVolatilityForEachChannel);
            var y = DataSet.ToYWorkingSet(tensorY, numClass, categoryByte=>categoryByte);
            return new InMemoryDataSet(x, y, "TestAbstractDataSet", Objective_enum.Classification, meanAndVolatilityForEachChannel);
        }
    }
}
