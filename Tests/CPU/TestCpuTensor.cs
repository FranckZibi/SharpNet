using System;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.Data;
using SharpNetTests.Datasets;
using SharpNetTests.NonReg;

namespace SharpNetTests.CPU
{
    [TestFixture]
    public class TestCpuTensor
    {
        [Test]
        public void TestEquals()
        {
            var a = new CpuTensor<float>(new []{10, 5});
            var b = new CpuTensor<float>(new[] { 10, 5 });
            var c = new CpuTensor<float>(new[] { 11, 5 });
            Assert.IsTrue(TestTensor.SameContent(a, b, 0.0));
            Assert.IsFalse(TestTensor.SameContent(a, c, 10));
            b[0] += 1e-4f;
            Assert.IsTrue(TestTensor.SameContent(a, b, 1e-3));
            Assert.IsFalse(TestTensor.SameContent(a, c, 1e-5));
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
            Assert.IsTrue(TestTensor.SameContent(expected, c, 1e-6));
            c = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            a = RandomFloatTensor(shape, rand, -maxValue, maxValue);
            x = RandomFloatTensor(new[] { 32, 1157, 1, 1 }, rand, -maxValue, maxValue);
            expected = new CpuTensor<float>(shape, null);
            for (int i = 0; i < expected.Count; ++i)
            {
                expected.SpanContent[i] = a.ReadonlyContent[i] * x.ReadonlyContent[i/(7*7)];
            }
            c.MultiplyTensor(a, x);
            Assert.IsTrue(TestTensor.SameContent(expected, c, 1e-6));
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
            Assert.IsTrue(TestTensor.SameContent(expectedLoss, predictedLoss, 1e-6));
        }

        [Test]
        public void TestComputeCategoricalCrossentropyWithHierarchyLoss()
        {
            var expected = GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = GetPredictedCategoricalCrossentropyWithHierarchy();
            var buffer = new CpuTensor<float>(new[]{expected.Shape[0]});
            var observedLoss = expected.ComputeLoss(predicted, NetworkConfig.LossFunctionEnum.CategoricalCrossentropyWithHierarchy, buffer);
            var expectedLossBuffer = new []
            {
                //no clue
                0f, 
                //star/pleine
                (float)-Math.Log(0.5), 
                //1 digit star
                (float)-Math.Log(0.3), 
                //star 5
                (float)-Math.Log(0.3)+(float)-Math.Log(0.25), 
                //star 16
                (float)-Math.Log(0.2)+(float)-Math.Log(0.35)+(float)-Math.Log(0.6), 
                //star *6
                (float)-Math.Log(0.2)+(float)-Math.Log(0.6),
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
            var acc = expected.ComputeAccuracy(predicted, NetworkConfig.LossFunctionEnum.CategoricalCrossentropyWithHierarchy, buffer);
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
                    Assert.IsTrue(TestTensor.SameContent(expectedDest, observedDest, 1e-6));
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
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-6));
        }

        [TestCase(100000, 0.5, false, 0, 0)] //when not training, dropout is disabled
        [TestCase(100000, 0.0, true, 0, 0)] // no 0 if drop probability = 0%
        [TestCase(100000, 1.0, true, 100000-10, 100000)]  // only 0 if drop probability = 100%
        [TestCase(100000, 0.25, true, (int)(100000*0.2), (int)(100000 *0.3))]
        [TestCase(100000, 0.75, true, (int)(100000 *0.7), (int)(100000 *0.8))]
        public void TestDropoutForward(int nbRows, double dropProbability, bool isTraining, int minEqualToZeroAfterDropout, int maxEqualToZeroAfterDropout)
        {
            var rand = new Random(0);
            var x = RandomFloatTensor(new []{nbRows, 1}, rand, 10, 20);
            var y = RandomFloatTensor(x.Shape, rand, 10, 20);
            var memoryPool = new TensorMemoryPool(null, false);
            var dropoutReserveSpace = memoryPool.GetFloatTensor(y.Shape);
            x.DropoutForward(y, dropProbability, isTraining, rand, dropoutReserveSpace, null);
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
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4 });
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, poolingSize, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingSize, stride, stride);
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 2, 2 }, new float[] { 333, 8, 14, 16, 22, 24, 30, 32, 38, 40, 46, 48 });
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));
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
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4});
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingHeight, poolingWidth, stride, stride));
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingHeight, poolingWidth, stride, stride);
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 2, 2}, new float[] { 333,7,13,15,21,23,29,31,37,39,45,47});
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));
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
        public void TestHuberGradient()
        {
            var yExpected = TestNetworkPropagation.FromNumpyArray("[[[0, 1], [0, 0]]]");
            var yPredicted = TestNetworkPropagation.FromNumpyArray("[[[0.6, 0.4], [0.4, 0.6]]]");
            var expectedGradient = TestNetworkPropagation.FromNumpyArray("[[[0.15, -0.15], [0.1, 0.15]]]");
            var observedGradient = new CpuTensor<float>(expectedGradient.Shape);
            observedGradient.HuberGradient(yExpected, yPredicted, 1f);
            TestTensor.SameContent(expectedGradient, observedGradient, 1e-6);
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
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, outputCPU, 1e-6));
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
            int categoryCount = result.Shape[1];
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                int indexFirstCategory = rand.Next(categoryCount);
                var expectedFirstCategory = (float)rand.NextDouble();
                result.Set(row, indexFirstCategory, expectedFirstCategory);
                int indexSecondCategory = (indexFirstCategory+7)%categoryCount;
                var expectedSecondCategory = 1f-expectedFirstCategory;
                result.Set(row, indexSecondCategory, expectedSecondCategory);
            }
            return result;
        }
    }
}
