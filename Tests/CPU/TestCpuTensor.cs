﻿using System;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNetTests.Datasets;
using SharpNetTests.NonReg;

namespace SharpNetTests.CPU
{
    [TestFixture]
    public class TestCpuTensor
    {
        private readonly Random _rand = new (0);

        [Test]
        public void TestNormalize()
        {
            var a = new CpuTensor<float>(new[] { 3, 3 }, new []{1f,2,3,-5,2,6,9,2,1 });
            var (aNormalized, _, _) = a.Normalize();
            var expected = new CpuTensor<float>(a.Shape, new [] { -0.0949158f, 0, -0.1324532f, -0.949158f, 0, 1.05962586f, 1.04407382f, 0, -0.9271726f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, aNormalized, 1e-5));
        }

        [Test]
        public void TestEquals()
        {
            var a = new CpuTensor<float>(new []{10, 5});
            var b = new CpuTensor<float>(new[] { 10, 5 });
            var c = new CpuTensor<float>(new[] { 11, 5 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(a, b, 0.0));
            Assert.IsFalse(TensorExtensions.SameFloatContent(a, c, 10));
            b[0] += 1e-4f;
            Assert.IsTrue(TensorExtensions.SameFloatContent(a, b, 1e-3));
            Assert.IsFalse(TensorExtensions.SameFloatContent(a, c, 1e-5));
        }
        /*
         *  1 2 3      
         *  4 5 6
         *  7 8 9
         * 
         */
        [Test]
        public void TestConvolution()
        {
            const int f = 3;
            var input = new CpuTensor<float>(new[] { 1, 1, 3, 3 }, new float[]{1,2,3,4,5,6,7,8,9});
            var convolution = new CpuTensor<float>(new[] { 1, 1, f, f }, new float[] { 1, 0, -1, 1, 0, -1, 1, 0, -1 });

            var padding = 0;
            var stride = 1;
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] { -6 });
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            padding = f/2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 1, 1, 3, 3}, new float[] { -7,-4,7,-15,-6,15,-13,-4,13 });
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            input = new CpuTensor<float>(new[] { 3, 1, 3, 3 });
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = -333;
            padding = 0;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 1, 1 }, new float[] {-340,-6, -6 });
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            padding = f / 2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 3, 3 }, new float[] { -7, -338, 7, -15, -340, 15, -13, -4, 13, -25, -4, 25, -42, -6, 42, -31, -4, 31, -43, -4, 43, -69, -6, 69, -49, -4, 49 });
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);
        }

        [Test]
        public void TestTranspose()
        {
            var src = new CpuTensor<float>(new[] { 1, 1 }, new [] { 1f });
            var output = new CpuTensor<float>(src.Shape);
            src.Transpose(output);
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, output, 1e-6));

            src = new CpuTensor<float>(new[] { 1, 6 }, new[] { 0f, 1f, 2f, 3f, 4f, 5f });
            output = new CpuTensor<float>(new[] { 6, 1 });
            src.Transpose(output);
            var expectedTranspose = new CpuTensor<float>(new[] { 6, 1 }, new[] { 0f, 1f, 2f, 3f, 4f, 5f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedTranspose, output, 1e-6));

            src = new CpuTensor<float>(new[] { 2, 3 }, new[] { 0f, 1f, 2f, 3f, 4f, 5f });
            output = new CpuTensor<float>(new[] { 3, 2 });
            src.Transpose(output);
            expectedTranspose = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0f, 3f, 1f, 4f, 2f, 5f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedTranspose, output, 1e-6));
        }


        [Test]
        public void TestTransposeSecondAndThirdDimension()
        {
            var data = Enumerable.Range(0, 2*3*4*5).Select(x => (float)x).ToArray();
            var X = new CpuTensor<float>(new[] { 2, 3, 4, 5 }, data);
            var observedTarget = X.Clone();
            X.TransposeSecondAndThirdDimension(observedTarget);
            var expectedTarget  = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[[0,1,2,3,4],[20,21,22,23,24],[40,41,42,43,44]],[[5,6,7,8,9],[25,26,27,28,29],[45,46,47,48,49]],[[10,11,12,13,14],[30,31,32,33,34],[50,51,52,53,54]],[[15,16,17,18,19],[35,36,37,38,39],[55,56,57,58,59]]],[[[60,61,62,63,64],[80,81,82,83,84],[100,101,102,103,104]],[[65,66,67,68,69],[85,86,87,88,89],[105,106,107,108,109]],[[70,71,72,73,74],[90,91,92,93,94],[110,111,112,113,114]],[[75,76,77,78,79],[95,96,97,98,99],[115,116,117,118,119]]]], numpy.float)");
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedTarget, observedTarget, 1e-6));
        }


        [Test]
        public void TestUpdateWithPositionalEncoding_AttnIsAllYouNeed()
        {
            var X = new CpuTensor<float>(new[] { 2, 3, 4 });
            X.ZeroMemory();
            X.UpdateWithPositionalEncoding_AttnIsAllYouNeed(100);
            var expectedTarget = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[0,1,0,1],[0.841470957,0.540302277,0.0998334214,0.995004177],[0.909297407,-0.416146845,0.198669329,0.980066597]],[[0,1,0,1],[0.841470957,0.540302277,0.0998334214,0.995004177],[0.909297407,-0.416146845,0.198669329,0.980066597]]], numpy.float)");
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedTarget, X, 1e-6));
        }


        [Test]
        public void TestSetToZeroAllElementsBelowMainDiagonal()
        {
            var src = new CpuTensor<float>(new[] { 1, 1 }, new[] { 1f });
            src.SetToZeroAllElementsBelowMainDiagonal();
            var expected = new CpuTensor<float>(src.Shape, new[] { 1f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            var data = new[] { 0.5f, 1f, 2f, 3f, 4f, 5f };
            src = new CpuTensor<float>(new[] { 1, 6 }, (float[])data.Clone());
            src.SetToZeroAllElementsBelowMainDiagonal();
            expected = new CpuTensor<float>(src.Shape, (float[])data.Clone());
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 6, 1 }, (float[])data.Clone());
            src.SetToZeroAllElementsBelowMainDiagonal();
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, 0, 0, 0, 0, 0});
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 2, 3 }, (float[])data.Clone());
            src.SetToZeroAllElementsBelowMainDiagonal();
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, 1f, 2f, 0f, 4f, 5f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 3, 2 }, (float[])data.Clone());
            src.SetToZeroAllElementsBelowMainDiagonal();
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, 1f, 0f, 3f, 0f, 0f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0.5f, 1, 2, 3, 4, 5, 6, 7, 8 });
            src.SetToZeroAllElementsBelowMainDiagonal();
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, 1, 2, 0, 4, 5, 0, 0, 8 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));
        }

        [Test]
        public void TestSetAllElementsAboveMainDiagonal()
        {
            const float valueForElementsAboveMainDiagonal = -666;
            var src = new CpuTensor<float>(new[] { 1, 1 }, new[] { 1f });
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            var expected = new CpuTensor<float>(src.Shape, new[] { 1f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            var data = new[] { 0.5f, 1f, 2f, 3f, 4f, 5f };
            src = new CpuTensor<float>(new[] { 1, 6 }, (float[])data.Clone());
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 6, 1 }, (float[])data.Clone());
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            expected = new CpuTensor<float>(src.Shape, (float[])data.Clone());
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 2, 3 }, (float[])data.Clone());
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal, 3f, 4f, valueForElementsAboveMainDiagonal });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 3, 2 }, (float[])data.Clone());
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, valueForElementsAboveMainDiagonal, 2f, 3f, 4f, 5f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0.5f, 1, 2, 3, 4, 5, 6, 7, 8 });
            src.SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal);
            expected = new CpuTensor<float>(src.Shape, new[] { 0.5f, valueForElementsAboveMainDiagonal, valueForElementsAboveMainDiagonal, 3, 4, valueForElementsAboveMainDiagonal, 6, 7, 8 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));
        }


        [Test]
        public void TestSetIdentityMatrix()
        {
            var rand = new Random(0);
            const double maxValue = 10.0;
            var src = RandomFloatTensor(new []{1, 1}, rand, -maxValue, maxValue);
            src.SetIdentityMatrix();
            var expected = new CpuTensor<float>(src.Shape, new[] { 1f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));

            src = RandomFloatTensor(new[] { 3, 3 }, rand, -maxValue, maxValue);
            src.SetIdentityMatrix();
            expected = new CpuTensor<float>(src.Shape, new[] { 1f,0,0,0,1,0,0,0,1 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expected, 1e-6));
        }


        [Test]
        public void TestMultiplyTensor()
        {
            var rand = new Random(0);
            var shape = new[] {32, 1157, 7, 7};
            const double maxValue = 10.0;
            var c = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            var a = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            var x = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            var expected = new CpuTensor<float>(shape, null);
            for (int i = 0; i < expected.Count; ++i)
            {
                expected.SpanContent[i] = a.ReadonlyContent[i]*x.ReadonlyContent[i];
            }
            c.MultiplyTensor(a, x);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, c, 1e-6));
            c = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            a = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            x = RandomFloatTensor(new[] { 32, 1157, 1, 1 }, rand, -maxValue, maxValue);
            expected = new CpuTensor<float>(shape, null);
            for (int i = 0; i < expected.Count; ++i)
            {
                expected.SpanContent[i] = a.ReadonlyContent[i] * x.ReadonlyContent[i/(7*7)];
            }
            c.MultiplyTensor(a, x);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, c, 1e-6));
        }



        [Test]
        public void TestInsertAtColumnIndex()
        {
            var source = new CpuTensor<float>(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6});
            var toAdd = new CpuTensor<float>(new[] { 2, 2 }, new float[] { 100, 200, 300, 400});
            
            var result = CpuTensor<float>.InsertAtColumnIndex(source, toAdd, 1);
            var expectedResult = new CpuTensor<float>(new[] { 2, 5 }, new float[] { 1, 100, 200, 2, 3, 4, 300, 400, 5, 6});
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));

            result = CpuTensor<float>.InsertAtColumnIndex(source, toAdd, 3);
            expectedResult = new CpuTensor<float>(new[] { 2, 5 }, new float[] { 1, 2, 3, 100, 200, 4, 5, 6, 300, 400});
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));

            result = CpuTensor<float>.InsertAtColumnIndex(source, toAdd, 0);
            expectedResult = new CpuTensor<float>(new[] { 2, 5 }, new float[] { 100, 200, 1, 2, 3, 300, 400, 4, 5, 6});
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));
        }

        [Test]
        public void TestInsertAtRowIndex()
        {
            var source = new CpuTensor<float>(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var toAdd = new CpuTensor<float>(new[] { 2, 3 }, new float[] { 100, 200, 300, 400, 500, 600 });

            var result = CpuTensor<float>.InsertAtRowIndex(source, toAdd, 0);
            var expectedResult = new CpuTensor<float>(new[] { 4, 3 }, new float[] { 100, 200, 300, 400, 500, 600, 1, 2, 3, 4, 5, 6 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));

            result = CpuTensor<float>.InsertAtRowIndex(source, toAdd, 1);
            expectedResult = new CpuTensor<float>(new[] { 4, 3 }, new float[] { 1, 2, 3, 100, 200, 300, 400, 500, 600, 4, 5, 6 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));

            result = CpuTensor<float>.InsertAtRowIndex(source, toAdd, 2);
            expectedResult = new CpuTensor<float>(new[] { 4, 3 }, new float[] { 1, 2, 3, 4, 5, 6, 100, 200, 300, 400, 500, 600 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));
        }

        [Test]
        public void TestLoadColumnsFromSource()
        {
            var toUpdate = new CpuTensor<float>(new[] { 2, 4 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var source = new CpuTensor<float>(new[] { 2, 4 }, new float[] { 101, 102, 103, 104, 105, 106, 107, 108 });

            var toUpdateCopy = (CpuTensor<float>)toUpdate.Clone();
            toUpdateCopy.LoadColumnsFromSource(source, new int[] { });
            Assert.IsTrue(TensorExtensions.SameFloatContent(toUpdate, toUpdateCopy, 1e-6));

            toUpdate.LoadColumnsFromSource(source, new[]{1,3});
            var expectedResult = new CpuTensor<float>(new[] { 2, 4 }, new float[] { 1, 102, 3, 104, 5, 106, 7, 108 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, toUpdate, 1e-6));
        }

        


        [Test]
        public void TestMergeHorizontally()
        {
            var source = new CpuTensor<float>(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var toAdd = new CpuTensor<float>(new[] { 2, 2 }, new float[] { 100, 200, 300, 400 });

            var result = CpuTensor<float>.MergeHorizontally(source, toAdd);
            var expectedResult = new CpuTensor<float>(new[] { 2, 5 }, new float[] { 1, 2, 3, 100, 200, 4, 5, 6, 300, 400 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, result, 1e-6));
        }
        

        [Test]
        public void TestChangeAxis()
        {
            var t = new CpuTensor<int>(new [] {2, 2, 2, 2}, Enumerable.Range(0, 16).ToArray());
            var t1 = (CpuTensor<int>)t.ChangeAxis(new[] {3, 2, 0, 1}).ChangeAxis(new[] {0, 1, 3, 2});
            var t2 = (CpuTensor<int>)t.ChangeAxis(new[] {3, 2, 1, 0});
            Assert.IsTrue(t1.ReadonlyContent.SequenceEqual(t2.ReadonlyContent));
            Assert.IsTrue(t1.ReadonlyContent.SequenceEqual(new []{ 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 }));
        }


        public static CpuTensor<float> GetExpectedCategoricalCrossentropyWithHierarchy()
        {
            var stars = TestCategoryHierarchy.StarExample();
            var allRows = new[] {
                stars.ExpectedPrediction(new string[] { }),                //no clue
                stars.ExpectedPrediction(new[] { "full" }),              //star/full
                stars.ExpectedPrediction(new[] { "1digit" }),              //1 digit star
                stars.ExpectedPrediction(new[] { "1digit", "5" }),         //star 5
                stars.ExpectedPrediction(new[] { "2digits", "1", "6" }),   //star 16
                stars.ExpectedPrediction(new[] { "2digits", "*", "6" })    //star *6
            };
            
            var expected = new CpuTensor<float>(new[] { allRows.Length, allRows[0].Length });
            for (var row = 0; row < allRows.Length; row++)
            {
                var p = allRows[row];
                for (int col = 0; col < p.Length; ++col)
                {
                    expected.Set(row, col, p[col]);
                }
            }

            return expected;
        }
        public static CpuTensor<float> GetPredictedCategoricalCrossentropyWithHierarchy()
        {
            int nbRows = GetExpectedCategoricalCrossentropyWithHierarchy().Shape[0];
            //we'll make the following prediction for all (6) elements
            // 20% sure it is star/2digits
            //      tens    = 35% 1 / 65% 3
            //      units   = 40% 2 / 60% 6
            // 50% sure is it star/full
            // 30% sure it is star/1digit
            //      25% 5 / 75% 6
            var singlePrediction = new[] { 30, 0.2f, 20, 30, 0.35f, 0, 0.65f, 100, 0, 0, 0.4f, 0, 0, 0, 0.6f, 0, 0, 0, 0.5f, 0.3f, 90, 0, 0, 0, 0, 0.25f, 0.75f, 0, 0, 0 };
            var predicted = new CpuTensor<float>(new []{ nbRows, 30});
            for(int row=0;row< nbRows; ++row)
            {
                for (int col = 0; col < predicted.Shape[1]; ++col)
                {
                    predicted.Set(row, col, singlePrediction[col]);
                }
            }
            return predicted;
        }



        [Test]
        public void TestComputeBackPropagationLossCategoricalCrossentropyWithHierarchy()
        {
            var expected = GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = GetPredictedCategoricalCrossentropyWithHierarchy();
            var expectedLossContent = new[]
            {
                //no clue
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                //star/full
                0,0.2f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5f,0.3f,0,0,0,0,0,0,0,0,0,0,
                //1 digit star
                0,0.2f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5f,-0.7f,0,0,0,0,0,0,0,0,0,0,
                //star 5
                0,0.2f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5f,-0.7f,0,0,0,0,0,-0.75f,0.75f,0,0,0,
                //star 16
                0,-0.8f,0,0,-0.65f,0,0.65f,0,0,0,0.4f,0,0,0,-0.4f,0,0,0,0.5f,0.3f,0,0,0,0,0,0,0,0,0,0,
                //star *6
                0,-0.8f,0,0,0,0,0,0,0,0,0.4f,0,0,0,-0.4f,0,0,0,0.5f,0.3f,0,0,0,0,0,0,0,0,0,0
            };
            var predictedLoss = new CpuTensor<float>(expected.Shape);
            predictedLoss.CategoricalCrossentropyWithHierarchyGradient(expected, predicted);
            var expectedLoss= new CpuTensor<float>(predictedLoss.Shape, expectedLossContent);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedLoss, predictedLoss, 1e-6));
        }


        [TestCase(new [] { 35f, 23, 47, 17, 10, 43, 9, 6, 28 }, new [] { 30f, 33, 45, 23, 8, 49, 12, 4, 31 }, 0.9f)]
        [TestCase(new []{86f,97,99,100,101,103,106,110,112,113},new [] {2f,20,28,27,50,29,7,17,6,12}, -0.1757575f)]
        public void Test_ComputeEvaluationMetric_SpearmanCorrelation(float[] y_true_array, float[] y_pred_array, float expected_value)
        {
            var y_true = CpuTensor<float>.New(y_true_array, 1);
            var y_pred = CpuTensor<float>.New(y_pred_array, 1);
            const EvaluationMetricEnum metric = EvaluationMetricEnum.SpearmanCorrelation;
            var buffer = new CpuTensor<float>(y_pred.ComputeMetricBufferShape(metric));
            var observedLoss = buffer.ComputeEvaluationMetric(y_true, y_pred, metric, null);
            Assert.AreEqual(expected_value, observedLoss, 1e-6);
        }

        [TestCase(new[] { 3, 5, 2.5f, 7 }, new[] { 2.5f, 5, 4, 8 }, 0.039730120450f)]
        public void Test_ComputeEvaluationMetric_MeanSquaredLogError(float[] y_true_array, float[] y_pred_array, float expected_value)
        {
            var y_true = CpuTensor<float>.New(y_true_array, 1);
            var y_pred = CpuTensor<float>.New(y_pred_array, 1);
            const EvaluationMetricEnum metric = EvaluationMetricEnum.MeanSquaredLogError;
            var buffer = new CpuTensor<float>(y_pred.ComputeMetricBufferShape(metric));
            var observedLoss = buffer.ComputeEvaluationMetric(y_true, y_pred, metric, null);
            Assert.AreEqual(expected_value, observedLoss, 1e-6);
        }

        [TestCase(new [] { 35f, 23, 47, 17, 10, 43, 9, 6, 28 }, new [] { 30f, 33, 45, 23, 8, 49, 12, 4, 31 }, 0.9498663391f)]
        [TestCase(new [] { 86f, 97, 99, 100, 101, 103, 106, 110, 112, 113 }, new [] { 2f, 20, 28, 27, 50, 29, 7, 17, 6, 12 }, -0.070216326029056739f)]
        public void Test_ComputeEvaluationMetric_PearsonCorrelation(float[] y_true_array, float[] y_pred_array, float expected_value)
        {
            var y_true = CpuTensor<float>.New(y_true_array, 1);
            var y_pred = CpuTensor<float>.New(y_pred_array, 1);
            const EvaluationMetricEnum metric = EvaluationMetricEnum.PearsonCorrelation;
            var buffer = new CpuTensor<float>(y_pred.ComputeMetricBufferShape(metric));
            var observedLoss = buffer.ComputeEvaluationMetric(y_true, y_pred, metric, null);
            Assert.AreEqual(expected_value, observedLoss, 1e-6);
        }

        [TestCase(new[] { 1f, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0 }, new[] { 0.29717186f, 0.9885438f, 0.64269745f, 0.7629636f, 0.030394293f, 0.3810045f, 0.34314185f, 0.95745516f, 0.5051292f, 0.7159725f, 0.11895773f, 0.27345148f, 0.90709794f, 0.7947656f, 0.33716035f, 0.45720878f, 0.14682505f, 0.22131474f, 0.41007328f, 0.7187268f, 0.6198303f, 0.48796806f, 0.19491343f, 0.8781915f, 0.8254232f, 0.7353975f, 0.8582192f, 0.679749f, 0.6248661f, 0.21840678f, 0.89536244f, 0.8964398f }, 0.4583333333333333f)]
        [TestCase(new[] { 1f, 1, 1, 0, 0 }, new[] { 0.9f, 0.8f, 0.45f, 0.55f, 0.65f }, 2/3f)]
        [TestCase(new[] { 1f, 1, 0 }, new[] { 0.9f, 0.8f, 0.85f }, 0.5f)]
        public void Test_ComputeEvaluationMetric_AUC(float[] y_true_array, float[] y_pred_array, float expected_value)
        {
            var y_true = CpuTensor<float>.New(y_true_array, 1);
            var y_pred = CpuTensor<float>.New(y_pred_array, 1);
            const EvaluationMetricEnum metric = EvaluationMetricEnum.AUC;
            var buffer = new CpuTensor<float>(y_pred.ComputeMetricBufferShape(metric));
            var observedLoss = buffer.ComputeEvaluationMetric(y_true, y_pred, metric, null);
            Assert.AreEqual(expected_value, observedLoss, 1e-6);
        }

        [TestCase(new[] { 0f, 0f, 1f, 1f }, new[] { 0.1f, 0.4f, 0.35f, 0.8f}, 0.8333333333333333f)]
        [TestCase(new[] { 0f, 0f, 0f, 0f }, new[] { 0.1f, 0.4f, 0.35f, 0.8f}, 0f)]
        [TestCase(new[] { 1f, 1f, 1f, 1f }, new[] { 0.1f, 0.4f, 0.35f, 0.8f}, 1f)]
        [TestCase(new[] { 0f, 0f, 1f, 0f, 1f, 1f, 0, 1f, 1f, 1f }, new[] { 0.65f, 0.1f, 0.15f, 0.43f, 0.97f, 0.24f, 0.82f, 0.7f, 0.32f, 0.84f }, 0.7688492063492063f)]
        public void Test_ComputeEvaluationMetric_AveragePrecisionScore(float[] y_true_array, float[] y_pred_array, float expected_value)
        {
            var y_true = CpuTensor<float>.New(y_true_array, 1);
            var y_pred = CpuTensor<float>.New(y_pred_array, 1);
            const EvaluationMetricEnum metric = EvaluationMetricEnum.AveragePrecisionScore;
            var buffer = new CpuTensor<float>(y_pred.ComputeMetricBufferShape(metric));
            var observedLoss = buffer.ComputeEvaluationMetric(y_true, y_pred, metric, null);
            Assert.AreEqual(expected_value, observedLoss, 1e-6);
        }

        [Test]
        public void TestComputeCategoricalCrossentropyWithHierarchyLoss()
        {
            var expected = GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = GetPredictedCategoricalCrossentropyWithHierarchy();
            var buffer = new CpuTensor<float>(new[]{expected.Shape[0]});
            var observedLoss = buffer.ComputeEvaluationMetric(expected, predicted, EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy, null);
            var expectedLossBuffer = new []
            {
                //no clue
                0f, 
                //star/pleine
                -MathF.Log(0.5f), 
                //1 digit star
                -MathF.Log(0.3f), 
                //star 5
                -MathF.Log(0.3f)+-MathF.Log(0.25f), 
                //star 16
                -MathF.Log(0.2f)+-MathF.Log(0.35f)+-MathF.Log(0.6f), 
                //star *6
                -MathF.Log(0.2f)+-MathF.Log(0.6f),
            };
            Assert.IsTrue(Utils.SameContent(expectedLossBuffer, buffer.ContentAsFloatArray(), 1e-6));
            var expectedLoss = expectedLossBuffer.Average();
            Assert.AreEqual(expectedLoss, observedLoss, 1e-6);
        }

        [Test]
        public void TestComputeAccuracy_CategoricalCrossentropyWithHierarchy()
        {
            var expected = GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = GetPredictedCategoricalCrossentropyWithHierarchy();
            var buffer = new CpuTensor<float>(new[] { expected.Shape[0] });
            var acc = buffer.ComputeAccuracyCategoricalCrossentropyWithHierarchy(expected, predicted);
            Assert.IsTrue(Utils.SameContent(new []{1f,1,0,0,0,0}, buffer.ContentAsFloatArray(), 1e-6));
            Assert.AreEqual(2.0/6, acc, 1e-6);
        }


        [Test]
        public void TestSoftmaxWithHierarchyActivation()
        {
            var root = TestCategoryHierarchy.StarExample();
            var x = GetPredictedCategoricalCrossentropyWithHierarchy();
            var observedActivation = new CpuTensor<float>(x.Shape);
            var rootPrediction = root.RootPrediction();
            var activationTensor = new CpuTensor<float>(new []{ rootPrediction.Length}, rootPrediction);
            CpuTensorActivationFunctions.SoftmaxWithHierarchy(x,observedActivation, activationTensor);
            var expectedActivationContent = new[]
            {
                //no clue6
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f,
                //star/full
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f,
                //1 digit star
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f,
                //star 5
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f,
                //star 16
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f,
                //star *6
                30f,0.28943312f,21f,30f,0.3273808f,0.23070136f,0.44191784f,100f,0.08838651f,0.08838651f,0.13185719f,0.08838651f,0.08838651f,0.08838651f,0.16105072f,0.08838651f,0.08838651f,0.08838651f,0.39069384f,0.31987306f,91f,0.09614436f,0.09614436f,0.09614436f,0.09614436f,0.12345181f,0.20353763f,0.09614436f,0.09614436f,0.09614436f
            };
            Assert.IsTrue(Utils.SameContent(expectedActivationContent, observedActivation.ContentAsFloatArray(), 1e-6));
        }

        

        [Test]
        public void TestZeroPadding()
        {
            var rand = new Random(0);
            foreach (var shape in new[] {new [] {7, 3, 7, 8}, new[] { 4, 5, 12, 5 } })
            {
                var src = RandomFloatTensor(shape, rand, -100.0, 100.0);
                foreach (var top_pad in new[] {0, 1, 3})
                foreach (var bottom_pad in new[] {0, 1, 3})
                foreach (var left_pad in new[] {0, 1, 3})
                foreach (var right_pad in new[] {0, 1, 3})
                {
                    var destShape = new[] { shape[0], shape[1], top_pad + shape[2] + bottom_pad, left_pad + shape[3] + right_pad};
                    var observedDest = RandomFloatTensor(destShape, rand, -100.0, 100.0);
                    observedDest.ZeroPadding(src, top_pad, bottom_pad, left_pad, right_pad);

                    var expectedDest = new CpuTensor<float>(destShape, null);
                    expectedDest.ZeroMemory();
                    for (int n = 0; n < shape[0]; ++n)
                    for (int c = 0; c < shape[1]; ++c)
                    for (int h = 0; h < shape[2]; ++h)
                    for (int w = 0; w < shape[3]; ++w)
                    {
                        expectedDest.Set(n, c, h + top_pad, w + left_pad, src.Get(n, c, h, w));
                    }
                    Assert.IsTrue(TensorExtensions.SameFloatContent(expectedDest, observedDest, 1e-6));
                }
            }
        }


        [Test]
        public void TestMultiplyEachRowIntoSingleValue()
        {
            var rand = new Random(0);
            var shape = new[] { 32, 1157, 7, 7 };
            const double maxValue = 10.0;
            var a = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            var b = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            var result = new CpuTensor<float>(new[] { 32, 1157, 1, 1 }, null);
            result.MultiplyEachRowIntoSingleValue(a,b);
            var expected = new CpuTensor<float>(result.Shape, null);
            for (int i = 0; i < a.Count; ++i)
            {
                expected.SpanContent[i/(7*7)] += a.ReadonlyContent[i] * b.ReadonlyContent[i];
            }
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-6));
        }

        [TestCase(100000, 0.5, false, 0, 0)] //when not training, dropout is disabled
        [TestCase(100000, 0.0, true, 0, 0)] // no 0 if drop probability = 0%
        [TestCase(100000, 1.0, true, 100000-10, 100000)]  // only 0 if drop probability = 100%
        [TestCase(100000, 0.25, true, (int)(100000*0.2), (int)(100000 *0.3))]
        [TestCase(100000, 0.75, true, (int)(100000 *0.7), (int)(100000 *0.8))]
        public void TestDropoutForward(int nbRows, double dropoutRate, bool isTraining, int minEqualToZeroAfterDropout, int maxEqualToZeroAfterDropout)
        {
            var rand = new Random(0);
            var x = RandomFloatTensor(new []{nbRows, 1}, rand, 10, 20);
            var y = RandomFloatTensor(x.Shape, rand, 10, 20);
            var memoryPool = new TensorMemoryPool(null);
            var dropoutReserveSpace = memoryPool.GetFloatTensor(y.Shape);
            x.DropoutForward(y, dropoutRate, isTraining, rand, dropoutReserveSpace);
            int nbObservedZeroAfterDropout = y.ReadonlyContent.Count(i => Math.Abs(i) < 1e-8);
            Assert.IsTrue(nbObservedZeroAfterDropout>=minEqualToZeroAfterDropout);
            Assert.IsTrue(nbObservedZeroAfterDropout<= maxEqualToZeroAfterDropout);
        }   

        [Test]
        public void TestMaxPooling4D()
        {
            const int poolingSize = 2;
            const int stride = 2;

            var input = new CpuTensor<float>(new[] { 1, 1, 3, 3 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, poolingSize, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingSize, stride, stride);
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] {5});
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4 });
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, poolingSize, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingSize, stride, stride);
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 2, 2 }, new float[] { 333, 8, 14, 16, 22, 24, 30, 32, 38, 40, 46, 48 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedOutput, output, 1e-6));
        }

        [Test]
        public void TestMaxPooling3D()
        {
            const int poolingHeight = 2;
            const int poolingWidth = 1;
            const int stride = 2;

            var input = new CpuTensor<float>(new[] { 1, 1, 3, 1}, new float[] { 1, 2, 3});
            var output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingHeight, poolingWidth, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingHeight, poolingWidth, stride, stride);
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] { 2 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4});
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingHeight, poolingWidth, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingHeight, poolingWidth, stride, stride);
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 2, 2}, new float[] { 333,7,13,15,21,23,29,31,37,39,45,47});
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedOutput, output, 1e-6));
        }

        [Test]
        public void TestSlice()
        {
            var data = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var owner = new CpuTensor<float>(new[] { 5, 2 }, data);
            var tensorTop2Rows = (CpuTensor<float>)owner.RowSlice(0, 2);  
            var tensorBottom3Rows = (CpuTensor<float>)owner.RowSlice(2, 3);
            var contentTop = tensorTop2Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[] { 0, 1, 2, 3 }, contentTop.ToArray());
            var contentBottom = tensorBottom3Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[] { 4, 5, 6, 7, 8, 9 }, contentBottom.ToArray());
            for (int i = 0; i < tensorTop2Rows.Count; ++i)
            {
                tensorTop2Rows.SpanContent[i] += 10;
            }
            for (int i = 0; i < tensorBottom3Rows.Count; ++i)
            {
                tensorBottom3Rows.SpanContent[i] += 20;
            }
            Assert.AreEqual(owner.ContentAsFloatArray(), new float[] { 10, 11, 12, 13, 24, 25, 26, 27, 28, 29 });
        }


        [Test]
        public void TestCosineSimilarityLoss()
        {
            var yExpected = TestNetworkPropagation.FromNumpyArray( "[3, 1, 2, 1, 0, 0, 5, 1]");
            var yPredicted = TestNetworkPropagation.FromNumpyArray("[1, 1, 0, 1, 0, 1, 0, 0]");
            var expectedLoss = TestNetworkPropagation.FromNumpyArray("[0.486664265, 0.66666667]");
            const int timeSeriesLength = 2;
            Debug.Assert(yPredicted.Count% timeSeriesLength == 0);
            var observedLoss = new CpuTensor<float>(new []{ timeSeriesLength });
            observedLoss.CosineSimilarityLossBuffer(yExpected, yPredicted, timeSeriesLength);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedLoss, observedLoss, 1e-6));
        }

        [Test]
        public void TestCosineSimilarityGradient()
        {
            //!D TODO 
            var yExpected = TestNetworkPropagation.FromNumpyArray("[[[0, 1], [0, 0]]]");
            var yPredicted = TestNetworkPropagation.FromNumpyArray("[[[0.6, 0.4], [0.4, 0.6]]]");
            var expectedGradient = TestNetworkPropagation.FromNumpyArray("[[[0.15, -0.15], [0.1, 0.15]]]");
            var observedGradient = new CpuTensor<float>(expectedGradient.Shape);
            observedGradient.CosineSimilarityGradient(yExpected, yPredicted, 2);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedGradient, observedGradient, 1e-6));
        }

        [Test]
        public void TestHuberGradient()
        {
            var yExpected = TestNetworkPropagation.FromNumpyArray("[[[0, 1], [0, 0]]]");
            var yPredicted = TestNetworkPropagation.FromNumpyArray("[[[0.6, 0.4], [0.4, 0.6]]]");
            var expectedGradient = TestNetworkPropagation.FromNumpyArray("[[[0.15, -0.15], [0.1, 0.15]]]");
            var observedGradient = new CpuTensor<float>(expectedGradient.Shape);
            const float huberDelta = 1f;
            observedGradient.HuberGradient(yExpected, yPredicted, huberDelta);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedGradient, observedGradient, 1e-6));
        }


        [TestCase(1, true)]
        [TestCase(1, false)]
        [TestCase(2, true)]
        [TestCase(2, false)]
        public void TestFocalLossWithZeroGammaAndBalancedDataset(int numClass, bool yExpectedHasOnly0And1)
        {
            // when the gamma of focal loss is 0 and the dataset is balanced, than focal loss should have the same value as binary crossentropy
            const int rows = 317;
            var metricConfig = new TestMetricConfig(bceWithFocalLossPercentageInTrueClass: 0.5f, bceWithFocalLoss_Gamma: 0f);
            var yExpected = yExpectedHasOnly0And1?RandomOneHotTensor(new[] { rows, numClass }, _rand):RandomFloatTensor(new[] { rows, numClass }, _rand, 0, 1);
            var yPredicted = RandomFloatTensor(yExpected.Shape, _rand, 0, 1);

            //we ensure that the loss is the same
            var bufferBinaryLoss = RandomTensor(new[] { rows });
            bufferBinaryLoss.ComputeLossBufferForEvaluationMetric(yExpected, yPredicted, yExpectedHasOnly0And1? EvaluationMetricEnum.BinaryCrossentropy: EvaluationMetricEnum.BCEContinuousY, metricConfig);
            var bufferFocalLoss = RandomTensor(new[] { rows });
            bufferFocalLoss.ComputeLossBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.BCEWithFocalLoss, metricConfig);
            TensorExtensions.SameFloatContent(bufferBinaryLoss, bufferFocalLoss, 1e-4, out var difference);
            Assert.IsTrue(string.IsNullOrEmpty(difference), difference);


            //we ensure that the gradient is the same
            yExpected = yExpectedHasOnly0And1 ? RandomOneHotTensor(new[] { rows, numClass }, _rand) : RandomFloatTensor(new[] { rows, numClass }, _rand, 0, 1);
            yPredicted = RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            var gradientsBinaryLoss = RandomTensor(yExpected.Shape);
            gradientsBinaryLoss.ComputeGradientForEvaluationMetric(yExpected, yPredicted, yExpectedHasOnly0And1 ? EvaluationMetricEnum.BinaryCrossentropy : EvaluationMetricEnum.BCEContinuousY, metricConfig);
            var gradientsFocalLoss = RandomTensor(yExpected.Shape);
            gradientsFocalLoss.ComputeGradientForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.BCEWithFocalLoss, metricConfig);
            TensorExtensions.SameFloatContent(gradientsBinaryLoss, gradientsFocalLoss, 1e-4, out difference);
            Assert.IsTrue(string.IsNullOrEmpty(difference), difference);
        }


        [TestCase(1, true)]
        [TestCase(1, false)]
        [TestCase(2, true)]
        [TestCase(2, false)]
        public void TestFocalLossGradientWithZeroGamma(int numClass, bool yExpectedHasOnly0And1)
        {
            // when the gamma of focal loss is 0, than focal loss should have the same value as binary crossentropy
            const int rows = 317;
            var metricConfig = new TestMetricConfig(bceWithFocalLossPercentageInTrueClass: 1f, bceWithFocalLoss_Gamma: 0f);
            var yExpected = yExpectedHasOnly0And1 ? RandomOneHotTensor(new[] { rows, numClass }, _rand) : RandomFloatTensor(new[] { rows, numClass }, _rand, 0, 1); var yPredicted = RandomFloatTensor(yExpected.Shape, _rand, 0, 1);

            var bufferBinaryGradient = RandomTensor(yExpected.Shape);
            bufferBinaryGradient.ComputeGradientForEvaluationMetric(yExpected, yPredicted, yExpectedHasOnly0And1 ? EvaluationMetricEnum.BinaryCrossentropy : EvaluationMetricEnum.BCEContinuousY, metricConfig);

            var bufferFocalGradient = RandomTensor(yExpected.Shape);
            bufferFocalGradient.ComputeGradientForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.BCEWithFocalLoss, metricConfig); TensorExtensions.SameFloatContent(bufferBinaryGradient, bufferFocalGradient, 1e-6, out var difference);
            Assert.IsTrue(string.IsNullOrEmpty(difference), difference);
        }



        [Test]
        public void TestWordEmbeddingForwardPropagation2D()
        {
            //x:                (batchSize, timeSteps)
            //y:                (batchSize, timeSteps, embedding_dim)
            //wordEmbedding:    (num_embeddings, embedding_dim)
            var x = TestNetworkPropagation.FromNumpyArray("[[3,1], [2,1], [2,3], [1,2]]");
            var wordEmbedding = TestNetworkPropagation.FromNumpyArray("[[0,0,0], [101,102,103], [201,202,203], [301,302,303]]");
            var yExpected = TestNetworkPropagation.FromNumpyArray("[ [[301,302,303],[101,102,103]], [ [201,202,203],[101,102,103]], [ [201,202,203],[301,302,303]], [[101,102,103],[201,202,203]] ]");
            var yPredicted = RandomTensor(yExpected.Shape);
            yPredicted.WordEmbeddingForwardPropagation(x, wordEmbedding, 0, 0, 0, 0);
            Assert.IsTrue(TensorExtensions.SameFloatContent(yExpected, yPredicted, 1e-6));
        }

        [Test]
        public void TestWordEmbeddingForwardPropagation3D()
        {
            //x:                (batchSize, timeSteps, inputSize)
            //y:                (batchSize, timeSteps, inputSize+embedding_dim-1)
            //wordEmbedding:    (num_embeddings, embedding_dim)
            var x = TestNetworkPropagation.FromNumpyArray("[ [[3000,3,3001],[1000,1,1001]], [[2000,2,2001],[1002,1,1003]], [[2002,2,2003],[3002,3,3003]], [[1003,1,1004],[2004,2,2005]] ]");
            var wordEmbedding = TestNetworkPropagation.FromNumpyArray("[[0,0,0], [101,102,103], [201,202,203], [301,302,303]]");
            var yExpected = TestNetworkPropagation.FromNumpyArray("[ [[3000,301,302,303,3001],[1000,101,102,103,1001]], [[2000,201,202,203,2001],[1002,101,102,103,1003]], [[2002,201,202,203,2003],[3002,301,302,303,3003]], [[1003,101,102,103,1004], [2004,201,202,203,2005]] ]");
            var yPredicted = RandomTensor(yExpected.Shape);
            const int xIndexInLastDimensionToUse = 1;
            yPredicted.WordEmbeddingForwardPropagation(x, wordEmbedding, xIndexInLastDimensionToUse, xIndexInLastDimensionToUse, xIndexInLastDimensionToUse, x.Shape[2]- xIndexInLastDimensionToUse-1);
            Assert.IsTrue(TensorExtensions.SameFloatContent(yExpected, yPredicted, 1e-6));
        }

        [Test]
        public void WordEmbeddingBackwardPropagation2D()
        {
            //x:                (batchSize, timeSteps)
            //dy:               (batchSize, timeSteps, embedding_dim)
            //wordEmbedding:    (num_embeddings, embedding_dim)
            var x = TestNetworkPropagation.FromNumpyArray("[[3,1], [2,1], [2,3], [1,2]]");
            var dxPredicted = RandomTensor(x.Shape);
            var dxExpected = TestNetworkPropagation.FromNumpyArray("[[0,0], [0,0], [0,0], [0,0]]");
            var dy = TestNetworkPropagation.FromNumpyArray("[ [[3.1,3.2,3.3],[1.1,1.2,1.3]], [ [2.1,2.2,2.3],[1.4,1.5,1.6]], [ [2.4,2.5,2.6],[3.4,3.5,3.6]], [[1.4,1.5,1.6],[2.7,2.8,2.9]] ]");
            var dwExpected = TestNetworkPropagation.FromNumpyArray("[[0,0,0], [3.9,4.2,4.5], [7.2,7.5,7.8], [6.5,6.7,6.9]]");
            var dwPredicted = RandomTensor(dwExpected.Shape);
            dwPredicted.WordEmbeddingBackwardPropagation(x, dxPredicted, dy, 0, 0, 0, 0);
            Assert.IsTrue(TensorExtensions.SameFloatContent(dwExpected, dwPredicted, 1e-6));
            Assert.IsTrue(TensorExtensions.SameFloatContent(dxExpected, dxPredicted, 1e-6));
        }

        [Test]
        public void WordEmbeddingBackwardPropagation3D()
        {
            //x:                (batchSize, timeSteps, inputSize)
            //dy:               (batchSize, timeSteps, inputSize+embedding_dim-1)
            //wordEmbedding:    (num_embeddings, embedding_dim)
            var x = TestNetworkPropagation.FromNumpyArray("[ [[3000,3,3001],[1000,1,1001]], [[2000,2,2001],[1002,1,1003]], [[2002,2,2003],[3002,3,3003]], [[1003,1,1004],[2004,2,2005]] ]");
            var dxPredicted = RandomTensor(x.Shape);
            var dxExpected = TestNetworkPropagation.FromNumpyArray("[ [[3.000,0,3.001],[1.000,0,1.001]], [[2.000,0,2.001],[1.002,0,1.003]], [[2.002,0,2.003],[3.002,0,3.003]], [[1.003,0,1.004],[2.004,0,2.005]] ]");
            var dy = TestNetworkPropagation.FromNumpyArray("[ [[3.000,3.1,3.2,3.3,3.001],[1.000,1.1,1.2,1.3,1.001]], [[2.000,2.1,2.2,2.3,2.001],[1.002,1.4,1.5,1.6,1.003]], [[2.002,2.4,2.5,2.6,2.003],[3.002,3.4,3.5,3.6,3.003]], [[1.003,1.7,1.8,1.9,1.004], [2.004,2.7,2.8,2.9,2.005]] ]");
            var dwExpected = TestNetworkPropagation.FromNumpyArray("[[0,0,0], [4.2,4.5,4.8], [7.2,7.5,7.8], [6.5,6.7,6.9]]");
            var dwPredicted = RandomTensor(dwExpected.Shape);
            const int xIndexInLastDimensionToUse = 1;
            dwPredicted.WordEmbeddingBackwardPropagation(x, dxPredicted, dy, xIndexInLastDimensionToUse, xIndexInLastDimensionToUse, xIndexInLastDimensionToUse, x.Shape[2]-xIndexInLastDimensionToUse-1);
            Assert.IsTrue(TensorExtensions.SameFloatContent(dwExpected, dwPredicted, 1e-6));
            Assert.IsTrue(TensorExtensions.SameFloatContent(dxExpected, dxPredicted, 1e-6));
        }



        [Test]
        public void TestQrFactorization()
        {
            //This test is coming from: https://rosettacode.org/wiki/QR_decomposition#C.23
            var A = new CpuTensor<float>(new[] { 5, 3}, new[] { 12.0f, -51, 4, 6, 167, -68, -4, 24, -41, -1, 1, 0, 2, 0, 3 });
            //var A = new CpuTensor<float>(new[] { 250, 10 });RandomFloatTensor(A.Shape, new Random(0), -1, 1).CopyTo(A);

            int m = A.Shape[0];
            int n = A.Shape[1];
            // the orthogonal 'Q' matrix of shape (m, n)
            var Q = new CpuTensor<float>(new[] { m, n }, null);
            // the upper triangular matrix 'R' of shape (n, n)
            var R = new CpuTensor<float>(new[] { n, n }, null);
            var floatBuffer = new CpuTensor<float>(new[] { A.QRFactorization_FloatBufferLength() }, null);
            A.QRFactorization(Q, R, floatBuffer);

            //var sw = System.Diagnostics.Stopwatch.StartNew();
            //int count = 1000;
            //for (int i = 0; i < count; ++i)
            //{
            //    A.QRFactorization(Q, R, floatBuffer);
            //}
            //Console.WriteLine("took " + (sw.Elapsed.TotalMilliseconds / count) + "ms");
            //return;


            var a_clone = A.Clone();
            a_clone.Dot(Q, R);
            Assert.IsTrue(TensorExtensions.SameFloatContent(a_clone, A, 1e-4));
            var maxError = Q.MaxErrorIfOrthogonalMatrix();
            Assert.IsTrue(Math.Abs(maxError)<1e-6);
        }


        //This test is coming from: https://en.wikipedia.org/wiki/QR_decomposition
        [TestCase("[[12,-51,4],[6,167,-68],[-4,24,-41]]", "[[0.857142866,-0.394285709,-0.331428587],[0.428571433,0.902857125,0.0342857353],[-0.285714298,0.171428576,-0.942857146]]")]
        //This test is coming from: https://rosettacode.org/wiki/QR_decomposition#C.23
        [TestCase("[[12,-51,4],[6,167,-68],[-4,24,-41],[-1,1,0],[2,0,3]]", " [[0.846414685,-0.391290814,-0.343124002],[0.423207343,0.904087186,0.0292699095],[-0.282138228,0.170420542,-0.932856023],[-0.0705345571,0.0140406527,0.00109936972],[0.141069114,-0.0166555103,0.105771616]]")]
        public void TestQ_Factorization(string input, string expectedOutput)
        {
            var A = TestNetworkPropagation.FromNumpyArray(input);
            var expected_Q = TestNetworkPropagation.FromNumpyArray(expectedOutput);
            var observed_Q = A.Clone();
            A.Q_Factorization(observed_Q);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected_Q, observed_Q, 1e-6), observed_Q.ToNumpy());
            A.Q_Factorization();
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected_Q, A, 1e-6), A.ToNumpy());

        }

        public static CpuTensor<float> RandomFloatTensor(int[] shape, Random rand, double minValue, double maxValue)
        {
            var content = new float[Utils.Product(shape)];
            Utils.UniformDistribution(content, rand, minValue, maxValue);
            return new CpuTensor<float>(shape, content);
        }
        public static CpuTensor<byte> RandomByteTensor(int[] shape, Random rand, byte minValue, byte maxValue)
        {
            var content = new byte[Utils.Product(shape)];
            Utils.UniformDistribution(content, rand, minValue, maxValue);
            return new CpuTensor<byte>(shape, content);
        }
        private static void TestStandardConvolution(CpuTensor<float> input, CpuTensor<float> convolution, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, CpuTensor<float> expectedOutput)
        {
            var outputCPU = new CpuTensor<float>(expectedOutput.Shape);
            input.Convolution(convolution, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, outputCPU, false, GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM, null);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedOutput, outputCPU, 1e-6));
        }
        public static CpuTensor<float> RandomOneHotTensor(int[] shape, Random rand)
        {
            var result = new CpuTensor<float>(shape);
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                result.Set(row, rand.Next(result.Shape[1]), 1f);
            }
            return result;
        }

        //random tensor
        //in each row: only 2 elements with non zero value, the sum of the 2 elements is always = 1.0
        public static CpuTensor<float> RandomTwoHotTensor(int[] shape, Random rand)
        {
            var result = new CpuTensor<float>(shape);
            int numClass = result.Shape[1];
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                int indexFirstCategory = rand.Next(numClass);
                var expectedFirstCategory = (float)rand.NextDouble();
                result.Set(row, indexFirstCategory, expectedFirstCategory);
                int indexSecondCategory = (indexFirstCategory+7)%numClass;
                var expectedSecondCategory = 1f-expectedFirstCategory;
                result.Set(row, indexSecondCategory, expectedSecondCategory);
            }
            return result;
        }

        //random tensor containing only integers values between minValueIncluded (included) and maxValueExcluded (excluded)
        //in each row: only 2 elements with non zero value, the sum of the 2 elements is always = 1.0
        public static CpuTensor<float> RandomIntValuesTensor(int[] shape, Random rand, int minValueIncluded, int maxValueExcluded)
        {
            var result = new CpuTensor<float>(shape);
            var resultSpan = result.SpanContent;
            for (int i = 0; i < resultSpan.Length; ++i)
            {
                resultSpan[i] = rand.Next(minValueIncluded, maxValueExcluded);
            }
            return result;
        }

        private CpuTensor<float> RandomTensor(int[] shape)
        {
            return RandomFloatTensor(shape, _rand, -1.5, +1.5);
        }

    }
}
