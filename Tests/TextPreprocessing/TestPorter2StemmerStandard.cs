using NUnit.Framework;
using Porter2StemmerStandard;

namespace SharpNetTests.TextPreprocessing;

[TestFixture]
public class TestPorter2StemmerStandard
{
    // ReSharper disable once IdentifierTypo
    private readonly EnglishPorter2Stemmer _stemmer = new EnglishPorter2Stemmer();


    [Test]
    public void TestStem()
    {
        Assert.AreEqual("abc", _stemmer.Stem("ABC").Value);
        Assert.AreEqual("cat", _stemmer.Stem("Cats").Value);
        Assert.AreEqual("provid", _stemmer.Stem("providing").Value);
        Assert.AreEqual("walk", _stemmer.Stem("Walked").Value);
        Assert.AreEqual("walk", _stemmer.Stem("Walks").Value);
        Assert.AreEqual("want", _stemmer.Stem("WANTED").Value);
        Assert.AreEqual("want", _stemmer.Stem("WANTing").Value);
        Assert.AreEqual("177", _stemmer.Stem("177").Value);
        Assert.AreEqual(" 17wop7 ", _stemmer.Stem(" 17WoP7 ").Value);
        Assert.AreEqual("walked.", _stemmer.Stem("WalkED.").Value);
        Assert.AreEqual(".walk", _stemmer.Stem(".WalkED").Value);
        Assert.AreEqual("i walked wait", _stemmer.Stem("I Walked Waiting").Value);

        //Assert.AreEqual("spec’ed", _stemmer.Stem("spec’ed").Value);

    }


}