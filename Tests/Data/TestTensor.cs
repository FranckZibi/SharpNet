using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNetTests.CPU;

namespace SharpNetTests.Data
{
    [TestFixture]
    public class TestTensor
    {
        private static GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);


        [Test]
        public void TestSerialization()
        {
            //double test
            var shape = new[] {10, 5};
            var rand = new Random(0);
            Tensor a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5, "a");
            var aSerialized = new Serializer().Add(a).ToString();
            var aDeserialized = (Tensor)Serializer.Deserialize(aSerialized, null)["a"];
            Assert.IsTrue(SameContent(a, aDeserialized, 1e-9));
            Tensor aGpu = a.ToGPU<float>(GpuWrapper);
            var aGpuSerialized = new Serializer().Add(aGpu).ToString();
            var aGpuDeserialized = (Tensor)Serializer.Deserialize(aGpuSerialized, GpuWrapper)["a"];
            Assert.IsTrue(SameContent(aGpu, aGpuDeserialized, 1e-9));

            //float test
            a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5, "a");
            aSerialized = new Serializer().Add(a).ToString();
            aDeserialized = (Tensor)Serializer.Deserialize(aSerialized, null)["a"];
            Assert.IsTrue(SameContent(a, aDeserialized, 1e-5));
            aGpu = a.ToGPU<float>(GpuWrapper);
            aGpuSerialized = new Serializer().Add(aGpu).ToString();
            aGpuDeserialized = (Tensor)Serializer.Deserialize(aGpuSerialized, GpuWrapper)["a"];
            Assert.IsTrue(SameContent(aGpu,aGpuDeserialized, 1e-5));
        }

        [Test]
        public void TestClone()
        {
            //float test
            var shape = new[] { 10, 5 };
            var rand = new Random(0);
            Tensor a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5, "a");
            var aCloned = a.Clone(null);
            Assert.IsTrue(SameContent(a, aCloned, 1e-5));
            var aGpu = a.ToGPU<float>(GpuWrapper);
            aCloned = aGpu.Clone(GpuWrapper);
            Assert.IsTrue(SameContent(aGpu, aCloned, 1e-5));
        }

        [Test]
        public void TestStandardConvolutionOutputShape()
        {
            var batchSize = 666;
            var inputChannels = 1313;
            var filtersCount = 317;

            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var f in new[] { 3, 5, 7 })
                    {
                        var inputShape = new[] { batchSize, inputChannels, h, w };
                        var shapeConvolution = new[] { filtersCount, inputChannels, f, f };
                        // padding = 0 , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 1, false).SequenceEqual(new[] { batchSize, filtersCount, h - f + 1, w - f + 1 }));
                        // padding = same , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 1, false).SequenceEqual(new[] { batchSize, filtersCount, h, w }));
                        // padding = 0, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 2, false).SequenceEqual(new[] { batchSize, filtersCount, (h - f) / 2 + 1, (w - f) / 2 + 1 }));
                        // padding = same, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 2, false).SequenceEqual(new[] { batchSize, filtersCount, (h - 1) / 2 + 1, (w - 1) / 2 + 1 }));
                    }
                }
            }
        }
        [Test]
        public void TestDepthwiseConvolutionOutputShape()
        {
            var batchSize = 666;
            var inputChannels = 1313;
            var depthMultiplier = 1;
            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var f in new[] { 3, 5, 7 })
                    {
                        var inputShape = new[] { batchSize, inputChannels, h, w };
                        var shapeConvolution = new[] { depthMultiplier, inputChannels, f, f };
                        // padding = 0 , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 1, true).SequenceEqual(new[] { batchSize, inputChannels, h - f + 1, w - f + 1 }));
                        // padding = same , stride == 1
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 1, true).SequenceEqual(new[] { batchSize, inputChannels, h, w }));
                        // padding = 0, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.VALID, 2, true).SequenceEqual(new[] { batchSize, inputChannels, (h - f) / 2 + 1, (w - f) / 2 + 1 }));
                        // padding = same, stride == 2
                        Assert.IsTrue(ConvolutionLayer.OutputShape(inputShape, shapeConvolution, ConvolutionLayer.PADDING_TYPE.SAME, 2, true).SequenceEqual(new[] { batchSize, inputChannels, (h - 1) / 2 + 1, (w - 1) / 2 + 1 }));
                    }
                }
            }
        }
        [Test]
        public void TestPoolingOutputShape()
        {
            var batchSize = 666;
            var channelDepth = 1313;

            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var poolingHeight in new[] { 2, 3, 5, 7, 8 })
                    { 
                        foreach (var poolingWidth in new[] { 2, 3, 5, 7, 8 })
                        {
                            var inputShape = new[] { batchSize, channelDepth, h, w };
                            //stride == 1
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape, poolingHeight, poolingWidth, 1).SequenceEqual(new[] { batchSize, channelDepth, h - poolingHeight + 1, w - poolingWidth+ 1 }));
                            // stride == 2
                            Assert.IsTrue(PoolingLayer.PoolingOutputShape(inputShape, poolingHeight, poolingWidth, 2).SequenceEqual(new[] { batchSize, channelDepth, (h - poolingHeight) / 2 + 1, (w - poolingWidth) / 2 + 1 }));
                        }
                    }
                }
            }
        }
        [Test]
	    public void TestWidthHeightDimension()
	    {
	        var t = new CpuTensor<int>(new[] {2, 3, 5}, "t");
	        Assert.AreEqual(3, t.Dimension);
	        Assert.AreEqual(30, t.Count);
	        Assert.AreEqual(2, t.Shape[0]);
	        Assert.AreEqual(3, t.Shape[1]);
	        Assert.AreEqual(5, t.Shape[2]);

	        t = new CpuTensor<int>(new[] { 7 }, "t");
	        Assert.AreEqual(1, t.Dimension);
	        Assert.AreEqual(7, t.Count);
	        Assert.AreEqual(7, t.Shape[0]);
        }

        public static bool SameContent(Tensor a, Tensor b, double epsilon)
        {
            if (!a.SameShape(b))
            {
                return false;
            }
            var aFloatContent = a.ContentAsFloatArray();
            var bFloatContent = b.ContentAsFloatArray();
            for (int i = 0; i < a.Count; ++i)
            {
                if (Math.Abs(aFloatContent[i] - bFloatContent[i]) > epsilon)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
