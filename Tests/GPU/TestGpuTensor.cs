using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNetTests.Data;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class TestGpuTensor
    {
        private static GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);
        [Test]
        public void TestDot()
        {
            // [1x1] x [1x1] matrix
            float[] data = new[] { 2f};
            var a = new GPUTensor<float>(new[] { 1, 1 }, data, GpuWrapper, "a");
            float[] data1 = new[] { 5f };
            var b = new GPUTensor<float>(new[] { 1, 1 }, data1, GpuWrapper, "b");
            var result = new GPUTensor<float>(new[] { 1, 1 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            var expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 10f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x2] x [2x2] matrix
            float[] data2 = new []{1f,2, 3, 4 };
            a = new GPUTensor<float>(new []{2,2}, data2, GpuWrapper, "a");
            float[] data3 = new []{1f,2, 3, 4 };
            b = new GPUTensor<float>(new []{2,2}, data3, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 2, 2}, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 7f, 10f, 15f, 22f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [1x2] x [2x1] matrix
            float[] data4 = new[] { 1f, 2};
            a = new GPUTensor<float>(new[] { 1, 2 }, data4, GpuWrapper, "a");
            float[] data5 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 2, 1 }, data5, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 1, 1 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 11f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x1] x [1x2] matrix
            float[] data6 = new[] { 1f, 2 };
            a = new GPUTensor<float>(new[] { 2, 1 }, data6, GpuWrapper, "a");
            float[] data7 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 1, 2 }, data7, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 2, 2 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected,result, 1e-9));

            // [2x1x1] x [1x1x2] matrix
            float[] data8 = new[] { 1f, 2 };
            a = new GPUTensor<float>(new[] { 2, 1,1 }, data8, GpuWrapper, "a");
            float[] data9 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 1, 1,2 }, data9, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 2, 2 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [10x1x28x28] x [784x100] matrix
            a = new GPUTensor<float>(new[] { 10, 1, 28,28 }, null, GpuWrapper, "a");
            b = new GPUTensor<float>(new[] { 784, 100 }, null, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 10, 1,1,100 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 10, 1,1,100 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [35x124] x [124x21] matrix
            var data10 = new float[35*124];
            a = new GPUTensor<float>(new[] { 35, 124 }, data10, GpuWrapper, "a");
            var data11 = new float[124*21];
            b = new GPUTensor<float>(new[] { 124, 21}, data11, GpuWrapper, "b");
            result = new GPUTensor<float>(new[] { 35, 21 }, null, GpuWrapper, "result");
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 35, 21}, new float[35*21], "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));
        }

        [Test]
        public void TestOwner()
        {
            var data = new float[]{0,1,2,3,4,5,6,7,8,9};
            var owner = new GPUTensor<float>(new []{5,2}, data, GpuWrapper, "owner");
            var tensorTop2Rows = new GPUTensor<float>(new[] { 2, 2}, owner, 0, "tensorTop2Rows");
            var tensorBottom3Rows = new GPUTensor<float>(new[] { 3, 2 }, owner, 4* owner.TypeSize, "tensorBottom3Rows");
            var contentTop = tensorTop2Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[]{0,1,2,3}, contentTop.ToArray());
            var contentBottom = tensorBottom3Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[] { 4, 5, 6, 7, 8, 9 }, contentBottom.ToArray());
        }
    }
}
