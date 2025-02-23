using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;
using SharpNetTests.CPU;

namespace SharpNetTests.Data
{
    [TestFixture]
    public class TestTensor
    {
        [Test]
        public void TestSerialization()
        {
            //double test
            var shape = new[] {10, 5};
            var rand = new Random(0);
            Tensor a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5);
            var aSerialized = new Serializer().Add("a", a).ToString();
            var aDeserialized = (Tensor)Serializer.Deserialize(aSerialized)["a"];
            Assert.IsTrue(TensorExtensions.SameFloatContent(a, aDeserialized, 1e-9));

            //float test
            a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5);
            aSerialized = new Serializer().Add("a", a).ToString();
            aDeserialized = (Tensor)Serializer.Deserialize(aSerialized)["a"];
            Assert.IsTrue(TensorExtensions.SameFloatContent(a, aDeserialized, 1e-5));
        }

        [Test]
        public void TestFillMinusOneIfAny()
        {
            int[] originalShape = { 4, 3, 5};
            Assert.AreEqual(new[] { 4, 3, 5 }, Tensor.FillMinusOneIfAny(originalShape, new[] { 4, 3, 5 }));
            Assert.AreEqual(new[] { 2, 2, 3, 5 }, Tensor.FillMinusOneIfAny(originalShape, new[] { 2, 2, 3, 5 }));
            Assert.AreEqual(new[]{4*3*5} ,Tensor.FillMinusOneIfAny(originalShape, new[]{-1}));
            Assert.AreEqual(new[] { 30, 2}, Tensor.FillMinusOneIfAny(originalShape, new[] { 30, -1 }));
            Assert.AreEqual(new[] { 2, 3, 10}, Tensor.FillMinusOneIfAny(originalShape, new[] { 2, 3, -1 }));

            Assert.Throws<ArgumentException>(() => Tensor.FillMinusOneIfAny(originalShape, new[] { -1,7 }));
            Assert.Throws<ArgumentException>(() => Tensor.FillMinusOneIfAny(originalShape, new[] { -1,-1,5}));
        }


        [Test]
        public void TestStandardConvolutionOutputShape()
        {
            const int batchSize = 666;
            const int input_channels = 1313;
            const int out_channels = 317;

            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var f in new[] { 3, 5, 7 })
                    {
                        var inputShape = new[] { batchSize, input_channels, h, w };
                        var shapeConvolution = new[] { out_channels, input_channels, f, f };
                        // padding = 0 , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 1, false).SequenceEqual(new[] { batchSize, out_channels, h - f + 1, w - f + 1 }));
                        // padding = same , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 1, false).SequenceEqual(new[] { batchSize, out_channels, h, w }));
                        // padding = 0, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 2, false).SequenceEqual(new[] { batchSize, out_channels, (h - f) / 2 + 1, (w - f) / 2 + 1 }));
                        // padding = same, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 2, false).SequenceEqual(new[] { batchSize, out_channels, (h - 1) / 2 + 1, (w - 1) / 2 + 1 }));
                    }
                }
            }
        }
        [Test]
        public void TestDepthwiseConvolutionOutputShape()
        {
            const int batchSize = 666;
            const int input_channels = 1313;
            const int depth_multiplier = 1;
            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var f in new[] { 3, 5, 7 })
                    {
                        var inputShape = new[] { batchSize, input_channels, h, w };
                        var shapeConvolution = new[] { depth_multiplier, input_channels, f, f };
                        // padding = 0 , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 1, true).SequenceEqual(new[] { batchSize, input_channels, h - f + 1, w - f + 1 }));
                        // padding = same , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 1, true).SequenceEqual(new[] { batchSize, input_channels, h, w }));
                        // padding = 0, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 2, true).SequenceEqual(new[] { batchSize, input_channels, (h - f) / 2 + 1, (w - f) / 2 + 1 }));
                        // padding = same, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 2, true).SequenceEqual(new[] { batchSize, input_channels, (h - 1) / 2 + 1, (w - 1) / 2 + 1 }));
                    }
                }
            }
        }
        [Test]
        public void TestPoolingOutputShape4D()
        {
            const int batchSize = 666;
            const int channelDepth = 1313;

            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var poolingHeight in new[] { 2, 3, 5, 7, 8 })
                    { 
                        foreach (var poolingWidth in new[] { 2, 3, 5, 7, 8 })
                        {
                            var inputShape4D = new[] { batchSize, channelDepth, h, w };
                            //stride == 1
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape4D, poolingHeight, poolingWidth, 1, 1).SequenceEqual(new[] { batchSize, channelDepth, h - poolingHeight + 1, w - poolingWidth+ 1 }));
                            // stride == 2
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape4D, poolingHeight, poolingWidth, 2, 2).SequenceEqual(new[] { batchSize, channelDepth, (h - poolingHeight) / 2 + 1, (w - poolingWidth) / 2 + 1 }));

                            var inputShape3D = new[] { batchSize, h, w };
                            //stride == 1
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape3D, poolingHeight, poolingWidth, 1, 1).SequenceEqual(new[] { batchSize, h - poolingHeight + 1, w - poolingWidth + 1 }));
                            // stride == 2
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape3D, poolingHeight, poolingWidth, 2, 2).SequenceEqual(new[] { batchSize, (h - poolingHeight) / 2 + 1, (w - poolingWidth) / 2 + 1 }));
                        }
                    }
                }
            }
        }

        [Test]
	    public void TestWidthHeightDimension()
	    {
	        var t = new CpuTensor<int>(new[] {2, 3, 5});
	        Assert.AreEqual(3, t.Dimension);
	        Assert.AreEqual(30, t.Count);
	        Assert.AreEqual(2, t.Shape[0]);
	        Assert.AreEqual(3, t.Shape[1]);
	        Assert.AreEqual(5, t.Shape[2]);

	        t = new CpuTensor<int>(new[] { 7 });
	        Assert.AreEqual(1, t.Dimension);
	        Assert.AreEqual(7, t.Count);
	        Assert.AreEqual(7, t.Shape[0]);
        }

    }
}
