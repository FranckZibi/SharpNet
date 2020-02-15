﻿using System;
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

            //var categoryIndexToDescription = new string[nbCategories];
            //for (int i = 0; i < nbCategories; ++i)
            //{
            //    categoryIndexToDescription[i] = rand.Next(nbCategories).ToString();
            //}
            var elementIdToCategoryIndex = new int[tensorX.Shape[0]];
            for (int i = 0; i < elementIdToCategoryIndex.Length; ++i)
            {
                elementIdToCategoryIndex[i] = i % nbCategories;
            }
            string name = "TestAbstractDataSet";
            var meanAndVolatilityForEachChannel = tensorX.ComputeMeanAndVolatilityOfEachChannel(t => t);
            var x = AbstractDataSet.ToXWorkingSet(tensorX, meanAndVolatilityForEachChannel);
            var y = AbstractDataSet.ToYWorkingSet(tensorY, nbCategories, categoryByte=>categoryByte);
            return new InMemoryDataSet(x, y, elementIdToCategoryIndex, name, meanAndVolatilityForEachChannel);
        }
    }
}