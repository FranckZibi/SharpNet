namespace SharpNetTests
{
    static class Program
    {
        private static void Main()
        {
            //new TestGradient().TestGradientForDenseLayer(true, true);
            new NonReg.TestResNetCIFAR10().Test();
            //new NonReg.TestMNIST().Test();
            //new NonReg.TestCIFAR10().Test();
            //new NonReg.TestNetworkPropagation().TestParallelRunWithTensorFlow();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
        }
    }
}
