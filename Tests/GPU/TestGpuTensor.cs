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
            var a = NewFloatGPUTensor(new[] { 1, 1 }, new[] { 2f}, "a", GpuWrapper);
            var b = NewFloatGPUTensor(new[] { 1, 1 }, new[] { 5f }, "b", GpuWrapper);
            var result = new GPUTensor<float>(new[] { 1, 1 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            var expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 10f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x2] x [2x2] matrix
            a = NewFloatGPUTensor(new []{2,2}, new []{1f,2, 3, 4 }, "a", GpuWrapper );
            b = NewFloatGPUTensor(new []{2,2}, new []{1f,2, 3, 4 }, "b", GpuWrapper );
            result = new GPUTensor<float>(new[] { 2, 2}, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 7f, 10f, 15f, 22f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [1x2] x [2x1] matrix
            a = NewFloatGPUTensor(new[] { 1, 2 }, new[] { 1f, 2}, "a",  GpuWrapper);
            b = NewFloatGPUTensor(new[] { 2, 1 }, new[] { 3f, 4 }, "b", GpuWrapper);
            result = new GPUTensor<float>(new[] { 1, 1 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 11f }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x1] x [1x2] matrix
            a = NewFloatGPUTensor(new[] { 2, 1 }, new[] { 1f, 2 }, "a", GpuWrapper);
            b = NewFloatGPUTensor(new[] { 1, 2 }, new[] { 3f, 4 }, "b", GpuWrapper);
            result = new GPUTensor<float>(new[] { 2, 2 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected,result, 1e-9));

            // [2x1x1] x [1x1x2] matrix
            a = NewFloatGPUTensor(new[] { 2, 1,1 }, new[] { 1f, 2 }, "a", GpuWrapper);
            b = NewFloatGPUTensor(new[] { 1, 1,2 }, new[] { 3f, 4 }, "b", GpuWrapper);
            result = new GPUTensor<float>(new[] { 2, 2 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [10x1x28x28] x [784x100] matrix
            a = new GPUTensor<float>(new[] { 10, 1, 28,28 }, "a", GpuWrapper);
            b = new GPUTensor<float>(new[] { 784, 100 }, "b", GpuWrapper);
            result = new GPUTensor<float>(new[] { 10, 1,1,100 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 10, 1,1,100 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [35x124] x [124x21] matrix
            a = NewFloatGPUTensor(new[] { 35, 124 }, new float[35*124], "a", GpuWrapper);
            b = NewFloatGPUTensor(new[] { 124, 21}, new float[124*21], "b", GpuWrapper);
            result = new GPUTensor<float>(new[] { 35, 21 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 35, 21}, new float[35*21], "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));
        }

        private static GPUTensor<float> NewFloatGPUTensor(int[] shape, float[] data, string description, GPUWrapper gpuWrapper)
        {
            using (var m = new HostPinnedMemory<float>(data))
            {
                return new GPUTensor<float>(shape, m.Pointer, description, gpuWrapper);
            }
        }

        [Test]
        public void TestOwner()
        {
            var owner = NewFloatGPUTensor(new []{5,2}, new float[]{0,1,2,3,4,5,6,7,8,9}, "owner", GpuWrapper);
            var tensorTop2Rows = new GPUTensor<float>(owner, new[] { 2, 2}, 0, "tensorTop2Rows");
            var tensorBottom3Rows = new GPUTensor<float>(owner, new[] { 3, 2 }, 4* owner.TypeSize, "tensorBottom3Rows");
            var contentTop = tensorTop2Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[]{0,1,2,3}, contentTop);
            var contentBottom = tensorBottom3Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[] { 4, 5, 6, 7, 8, 9 }, contentBottom);
        }
    }
}
