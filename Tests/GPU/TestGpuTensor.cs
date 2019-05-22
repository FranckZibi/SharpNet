using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNetTests.Data;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class TestGpuTensor
    {
        private GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);
        [Test]
        public void TestDot()
        {
            // [1x1] x [1x1] matrix
            var a = NewDoubleGPUTensor(new[] { 1, 1 }, new[] { 2.0}, "a", GpuWrapper);
            var b = NewDoubleGPUTensor(new[] { 1, 1 }, new[] { 5.0 }, "b", GpuWrapper);
            var result = new GPUTensor<double>(new[] { 1, 1 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            var expected = new CpuTensor<double>(new[] { 1, 1 }, new[] { 10.0 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x2] x [2x2] matrix
            a = NewDoubleGPUTensor(new []{2,2}, new []{1.0,2.0, 3.0, 4.0 }, "a", GpuWrapper );
            b = NewDoubleGPUTensor(new []{2,2}, new []{1.0,2.0, 3.0, 4.0 }, "b", GpuWrapper );
            result = new GPUTensor<double>(new[] { 2, 2}, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 2, 2 }, new[] { 7.0, 10.0, 15.0, 22.0 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [1x2] x [2x1] matrix
            a = NewDoubleGPUTensor(new[] { 1, 2 }, new[] { 1.0, 2.0}, "a",  GpuWrapper);
            b = NewDoubleGPUTensor(new[] { 2, 1 }, new[] { 3.0, 4.0 }, "b", GpuWrapper);
            result = new GPUTensor<double>(new[] { 1, 1 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 1, 1 }, new[] { 11.0 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [2x1] x [1x2] matrix
            a = NewDoubleGPUTensor(new[] { 2, 1 }, new[] { 1.0, 2.0 }, "a", GpuWrapper);
            b = NewDoubleGPUTensor(new[] { 1, 2 }, new[] { 3.0, 4.0 }, "b", GpuWrapper);
            result = new GPUTensor<double>(new[] { 2, 2 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 2, 2 }, new[] { 3.0, 4.0, 6.0, 8.0 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected,result, 1e-9));

            // [2x1x1] x [1x1x2] matrix
            a = NewDoubleGPUTensor(new[] { 2, 1,1 }, new[] { 1.0, 2.0 }, "a", GpuWrapper);
            b = NewDoubleGPUTensor(new[] { 1, 1,2 }, new[] { 3.0, 4.0 }, "b", GpuWrapper);
            result = new GPUTensor<double>(new[] { 2, 2 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 2, 2 }, new[] { 3.0, 4.0, 6.0, 8.0 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [10x1x28x28] x [784x100] matrix
            a = new GPUTensor<double>(new[] { 10, 1, 28,28 }, "a", GpuWrapper);
            b = new GPUTensor<double>(new[] { 784, 100 }, "b", GpuWrapper);
            result = new GPUTensor<double>(new[] { 10, 1,1,100 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 10, 1,1,100 }, "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));

            // [35x124] x [124x21] matrix
            a = NewDoubleGPUTensor(new[] { 35, 124 }, new double[35*124], "a", GpuWrapper);
            b = NewDoubleGPUTensor(new[] { 124, 21}, new double[124*21], "b", GpuWrapper);
            result = new GPUTensor<double>(new[] { 35, 21 }, "result", GpuWrapper);
            result.Dot(a, false, b, false, 1.0, 0);
            expected = new CpuTensor<double>(new[] { 35, 21}, new double[35*21], "expected");
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-9));
        }

        private GPUTensor<double> NewDoubleGPUTensor(int[] shape, double[] data, string description, GPUWrapper gpuWrapper)
        {
            using (var m = new HostPinnedMemory<double>(data))
            {
                return new GPUTensor<double>(shape, m.Pointer, description, gpuWrapper);
            }
        }


        [Test]
        public void TestOwner()
        {
            var owner = NewDoubleGPUTensor(new []{5,2}, new double[]{0,1,2,3,4,5,6,7,8,9}, "owner", GpuWrapper);
            var tensorTop2Rows = new GPUTensor<double>(owner, new[] { 2, 2}, 0, "tensorTop2Rows");
            var tensorBottom3Rows = new GPUTensor<double>(owner, new[] { 3, 2 }, 4*sizeof(double), "tensorBottom3Rows");

            var contentTop = tensorTop2Rows.ContentAsDoubleArray();
            Assert.AreEqual(new double[]{0,1,2,3}, contentTop);
            var contentBottom = tensorBottom3Rows.ContentAsDoubleArray();
            Assert.AreEqual(new double[] { 4, 5, 6, 7, 8, 9 }, contentBottom);
        }

    }
}
