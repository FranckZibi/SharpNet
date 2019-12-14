using System;
using System.Collections.Generic;
using NUnit.Framework;
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
            var tensorX = TestCpuTensor.RandomByteTensor(shape, rand, 0, 255, "tensorX");
            var dataSet = GetRandomDataSet(tensorX, 2, rand);
            for (int elementId = 0; elementId < shape[0]; ++elementId)
            {
                var bmp = dataSet.OriginalElementContent(elementId);
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

        private static AbstractDataSet GetRandomDataSet(CpuTensor<byte> tensorX, int nbCategories, Random rand)
        {
            var tensorY = TestCpuTensor.RandomByteTensor(new[] { tensorX.Shape[0], 1 }, rand, 0, 1, "tensorY");

            var categoryIdToDescription = new string[nbCategories];
            for (int i = 0; i < nbCategories; ++i)
            {
                categoryIdToDescription[i] = rand.Next(nbCategories).ToString();
            }
            var elementIdToCategoryId = new int[tensorX.Shape[0]];
            for (int i = 0; i < elementIdToCategoryId.Length; ++i)
            {
                elementIdToCategoryId[i] = i % nbCategories;
            }
            string name = "TestAbstractDataSet";
            var meanAndVolatilityForEachChannel = tensorX.ComputeMeanAndVolatilityOfEachChannel(t => (float)t);
            ToWorkingSet(tensorX, tensorY, out CpuTensor<float> x, out CpuTensor<float> y, meanAndVolatilityForEachChannel);
            return new InMemoryDataSet(x, y, elementIdToCategoryId, categoryIdToDescription, name, meanAndVolatilityForEachChannel);
        }

        private static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet, List<Tuple<float, float>> meanAndVolatilityOfEachChannel)
        {
            xWorkingSet = x.Select((n, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            yWorkingSet = y.ToCategorical(1f, out _);
        }
    }
}
