using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.TextPreprocessing;
using SharpNetTests.Data;

namespace SharpNetTests.TextPreprocessing;

[TestFixture]
public class TestTfIdf
{
    [Test]
    public void Test1()
    {
        // each word is unique
        var observed = TfIdf.ToTfIdf(new List<string> { "a b c", "d e f" }, 6);
        var ln2 = (float)Math.Log(2);
        var expected = new CpuTensor<float>(new[] { 2, 6 }, new[] { ln2, ln2, ln2, 0,0,0, 0, 0, 0, ln2, ln2, ln2 });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));

        // one word is in all documents
        observed = TfIdf.ToTfIdf(new List<string> { "a b c", "a e f" }, 6);
        expected = new CpuTensor<float>(new[] { 2, 6 }, new[] { 0, ln2, ln2, 0, 0, 0, 0, 0, 0, ln2, ln2, 0 });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));

        observed = TfIdf.ToTfIdf(new List<string> { "a b c", "b c", "c" }, 3);
        expected = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0, (float)Math.Log(3 / 2.0), (float)Math.Log(3 / 1.0), 0, (float)Math.Log(3 / 2.0), 0, 0, 0, 0});

        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));


    }
}
