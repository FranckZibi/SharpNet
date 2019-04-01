using System;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.CPU;

namespace SharpNetTests.Data
{
    [TestFixture]
    public class TestTensor
    {
        private readonly GPUWrapper _gpuWrapper = GPUWrapper.Default;


        [Test]
        public void TestSerialization()
        {
            //double test
            var shape = new[] {10, 5};
            var rand = new Random(0);
            Tensor a = TestCpuTensor.RandomDoubleTensor(shape, rand, -1.5, +1.5, "a");
            var aSerialized = new Serializer().Add(a).ToString();
            var aDeserialized = (Tensor)Serializer.Deserialize(aSerialized, null)["a"];
            Assert.IsTrue(SameContent(a, aDeserialized, 1e-9));
            Tensor aGpu = a.ToGPU<double>(_gpuWrapper);
            var aGpuSerialized = new Serializer().Add(aGpu).ToString();
            var aGpuDeserialized = (Tensor)Serializer.Deserialize(aGpuSerialized, _gpuWrapper)["a"];
            Assert.IsTrue(SameContent(aGpu, aGpuDeserialized, 1e-9));

            //float test
            a = TestCpuTensor.RandomFloatTensor(shape, rand, -1.5, +1.5, "a");
            aSerialized = new Serializer().Add(a).ToString();
            aDeserialized = (Tensor)Serializer.Deserialize(aSerialized, null)["a"];
            Assert.IsTrue(SameContent(a, aDeserialized, 1e-5));
            aGpu = a.ToGPU<float>(_gpuWrapper);
            aGpuSerialized = new Serializer().Add(aGpu).ToString();
            aGpuDeserialized = (Tensor)Serializer.Deserialize(aGpuSerialized, _gpuWrapper)["a"];
            Assert.IsTrue(SameContent(aGpu,aGpuDeserialized, 1e-5));
        }


        [Test]
        public void TestConvolutionOutputShape()
        {
            var batchSize = 666;
            var channelDepth = 1313;
            var filtersCount = 317;

            foreach (var h in new[] { 3, 33, 50, 100, 200, 201 })
            {
                foreach (var w in new[] { 3, 33, 50, 100, 200, 201 })
                {
                    foreach (var f in new[] { 3, 5, 7 })
                    {
                        var shapeIntput = new[] { batchSize, channelDepth, h, w };
                        var shapeConvolution = new[] { filtersCount, channelDepth, f, f };
                        // padding = 0 , stride == 1
                        Assert.IsTrue(Tensor.ConvolutionOutputShape(shapeIntput, shapeConvolution, 0, 1).SequenceEqual(new[] { batchSize, filtersCount, h - f + 1, w - f + 1 }));
                        // padding = same , stride == 1
                        Assert.IsTrue(Tensor.ConvolutionOutputShape(shapeIntput, shapeConvolution, f / 2, 1).SequenceEqual(new[] { batchSize, filtersCount, h, w }));
                        // padding = 0, stride == 2
                        Assert.IsTrue(Tensor.ConvolutionOutputShape(shapeIntput, shapeConvolution, 0, 2).SequenceEqual(new[] { batchSize, filtersCount, (h - f) / 2 + 1, (w - f) / 2 + 1 }));
                        // padding = same, stride == 2
                        Assert.IsTrue(Tensor.ConvolutionOutputShape(shapeIntput, shapeConvolution, f / 2, 2).SequenceEqual(new[] { batchSize, filtersCount, (h - 1) / 2 + 1, (w - 1) / 2 + 1 }));
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
                    foreach (var poolingSizeForPoolingOutputShape in new[] { 2, 3, 5, 7, 8 })
                    {
                        var shapeIntput = new[] { batchSize, channelDepth, h, w };
                        //stride == 1
                        Assert.IsTrue(Tensor.PoolingOutputShape(shapeIntput, poolingSizeForPoolingOutputShape, 1).SequenceEqual(new[] { batchSize, channelDepth, h - poolingSizeForPoolingOutputShape + 1, w - poolingSizeForPoolingOutputShape + 1 }));
                        // stride == 2
                        Assert.IsTrue(Tensor.PoolingOutputShape(shapeIntput, poolingSizeForPoolingOutputShape, 2).SequenceEqual(new[] { batchSize, channelDepth, (h - poolingSizeForPoolingOutputShape) / 2 + 1, (w - poolingSizeForPoolingOutputShape) / 2 + 1 }));
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
	        Assert.AreEqual(2, t.Height);
	        Assert.AreEqual(3, t.Shape[1]);
	        Assert.AreEqual(5, t.Shape[2]);
	        Assert.AreEqual(3, t.Width);

	        t = new CpuTensor<int>(new[] { 7 }, "t");
	        Assert.AreEqual(1, t.Dimension);
	        Assert.AreEqual(7, t.Count);
	        Assert.AreEqual(7, t.Shape[0]);
	        Assert.AreEqual(7, t.Height);
	        Assert.AreEqual(1, t.Width);
        }

        public static bool SameContent(Tensor a, Tensor b, double epsilon)
        {
            if (!a.SameShape(b))
            {
                return false;
            }
            var aDoubleContent = a.ContentAsDoubleArray();
            var bDoubleContent = b.ContentAsDoubleArray();
            for (int i = 0; i < a.Count; ++i)
            {
                if (Math.Abs(aDoubleContent[i] - bDoubleContent[i]) > epsilon)
                {
                    return false;
                }
            }
            return true;
        }

    }
}
