using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.TextPreprocessing;
using SharpNetTests.Data;

namespace SharpNetTests.TextPreprocessing;

[TestFixture]
public class TestTfIdfEncoding
{
    [Test]
    public void TestEncode()
    {
        // each word is unique
        var observed = TfIdfEncoding.Encode(new List<string> { "a b c", "d e f" }, 6, "", norm:TfIdfEncoding.TfIdfEncoding_norm.None);
        var val0 = 0.231049061f;
        var expected = new CpuTensor<float>(new[] { 2, 6 }, new[] { val0, val0, val0, 0,0,0, 0, 0, 0, val0, val0, val0 });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));

        // one same word is in all documents
        observed = TfIdfEncoding.Encode(new List<string> { "a b c", "a e f" }, 6, "", norm: TfIdfEncoding.TfIdfEncoding_norm.None);
        expected = new CpuTensor<float>(new[] { 2, 6 }, new[] { 0, val0, val0, 0, 0, 0, 0, 0, 0, val0, val0, 0 });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));

        observed = TfIdfEncoding.Encode(new List<string> { "a b c", "b c", "c" }, 3, "", norm: TfIdfEncoding.TfIdfEncoding_norm.None);
        expected = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0,0.135155037f, 0.3662041f, 0, 0.202732548f, 0, 0, 0, 0 });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
    }

    [Test]
    public void TestEncodeV2()
    {
        //this is inspired by https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d

        //scikit-learn compatibility mode
        //no normalization
        var documents = new List<string> { "john cat", "cat eat fish", "eat big fish" };
        var observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.None, scikitLearnCompatibilityMode: true);
        var expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.643841f, 0f, 0f, 0.8465736f, 0f, 0.429227352f, 0.429227352f, 0.429227352f, 0f, 0f, 0f, 0.429227352f, 0.429227352f, 0f, 0.564382434f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
        // L1 normalization
        observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.L1, scikitLearnCompatibilityMode: true);
        expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.431987852f, 0f, 0f, 0.5680121f, 0f, 0.333333343f, 0.333333343f, 0.333333343f, 0f, 0f, 0f, 0.301670045f, 0.301670045f, 0f, 0.3966599f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
        // L2 normalization
        observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode: true);
        expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.605348468f, 0f, 0f, 0.795960546f, 0f, 0.577350259f, 0.577350259f, 0.577350259f, 0f, 0f, 0f, 0.517856061f, 0.517856061f, 0f, 0.6809186f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));

        //standard Tf-Idf
        //no normalization
        observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.None, scikitLearnCompatibilityMode: false);
        expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.202732548f, 0f, 0f, 0.549306154f, 0f, 0.135155037f, 0.135155037f, 0.135155037f, 0f, 0f, 0f, 0.135155037f, 0.135155037f, 0f, 0.3662041f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
        // L1 normalization
        observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.L1, scikitLearnCompatibilityMode: false);
        expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.2695773f, 0f, 0f, 0.730422735f, 0f, 0.3333333f, 0.3333333f, 0.3333333f, 0f, 0f, 0f, 0.212336242f, 0.212336242f, 0f, 0.575327456f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
        // L2 normalization
        observed = TfIdfEncoding.Encode(documents, 5, "", norm: TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode: false);
        expected = new CpuTensor<float>(new[] { 3, 5 }, new[] { 0.346241534f, 0f, 0f, 0.9381454f, 0f, 0.577350259f, 0.577350259f, 0.577350259f, 0f, 0f, 0f, 0.327184558f, 0.327184558f, 0f, 0.8865103f });
        Assert.IsTrue(TestTensor.SameContent(expected, observed.FloatCpuTensor(), 1e-5));
    }

}
