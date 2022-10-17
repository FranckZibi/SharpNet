using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet.TextPreprocessing;

namespace SharpNetTests.TextPreprocessing
{
    [TestFixture]
    public class TestTokenizer
    {
        [Test]
        public void TestExtractWords()
        {
            var tokenizer = new Tokenizer();
            Assert.IsTrue(new[] { "a", "bc", "def" }.SequenceEqual(tokenizer.ExtractWords("a\n|bC *  :def!?")));
            Assert.IsTrue(new string[] { }.SequenceEqual(tokenizer.ExtractWords("\n| *  :!?")));
        }

        [Test]
        public void TestFitOnTexts()
        {
            var tokenizer = new Tokenizer();
            tokenizer.FitOnTexts(new []{"z b c d", "z b c", "z b", "z"});
            Assert.IsTrue(new[] { "b", "c", "d", "z" }.SequenceEqual(tokenizer.WordIndex.Keys.OrderBy(x => x)));
            Assert.AreEqual(1 , tokenizer.WordIndex["z"]);
            Assert.AreEqual(2 , tokenizer.WordIndex["b"]);
            Assert.AreEqual(3 , tokenizer.WordIndex["c"]);
            Assert.AreEqual(4 , tokenizer.WordIndex["d"]);

            tokenizer = new Tokenizer(100, "<OOV>");
            tokenizer.FitOnTexts(new[] { "z b c d", "z b c", "z b", "z" });
            Assert.IsTrue(new[] { "<OOV>", "b", "c", "d", "z" }.SequenceEqual(tokenizer.WordIndex.Keys.OrderBy(x => x)));
            Assert.AreEqual(1, tokenizer.WordIndex["<OOV>"]);
            Assert.AreEqual(2, tokenizer.WordIndex["z"]);
            Assert.AreEqual(3, tokenizer.WordIndex["b"]);
            Assert.AreEqual(4, tokenizer.WordIndex["c"]);
            Assert.AreEqual(5, tokenizer.WordIndex["d"]);
        }

        [Test]
        public void TestTextToSequence()
        {
            var tokenizer = new Tokenizer();
            tokenizer.FitOnTexts(new[] { "z b c d", "b c z", "z b", "z" });
            Assert.IsTrue(new[] {1, 2,3,4}.SequenceEqual(tokenizer.TextToSequence("z b g c d f")));

            tokenizer = new Tokenizer(1000, "<OOV>");
            tokenizer.FitOnTexts(new[] { "z b c d", "b c z", "z b", "z" });
            Assert.IsTrue(new[] { 2, 3, 1, 4, 5, 1 }.SequenceEqual(tokenizer.TextToSequence("z b g c d f")));

            tokenizer = new Tokenizer(3+1, "<OOV>"); //only the 3 most common words +"<OOV>"
            tokenizer.FitOnTexts(new[] { "z b c d", "b c z", "z b", "z" });
            Assert.IsTrue(new[] { 2, 3, 1, 4, 1, 1 }.SequenceEqual(tokenizer.TextToSequence("z b g c d f")));

            tokenizer = new Tokenizer(2 + 1); //only the 3 most common words
            tokenizer.FitOnTexts(new[] { "z b c d", "b c z", "z b", "z" });
            Assert.IsTrue(new[] { 1, 2, 3}.SequenceEqual(tokenizer.TextToSequence("z b g c d f")));
        }

        [Test]
        public void TestSequenceToText()
        {
            var tokenizer = new Tokenizer();
            tokenizer.FitOnTexts(new[] {"z b c d", "b c z", "z b", "z"});
            Assert.AreEqual("z z d", tokenizer.SequenceToText(new List<int> {1, 1, 4}));
        }
    }
}
