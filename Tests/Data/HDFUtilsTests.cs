using NUnit.Framework;
using SharpNet.Data;

namespace SharpNetTests.Data
{
    [TestFixture]
    public class HDFUtilsTests
    {
        [Test]
        public void TestExtractString()
        {
            var data = new byte[] {(byte)'a', 0, 0, (byte)'b', (byte)'c', 0, (byte)'d', (byte)'e', (byte)'f' };
            Assert.AreEqual("a", HDFUtils.ExtractString(data, 0, 3));
            Assert.AreEqual("bc", HDFUtils.ExtractString(data, 1, 3));
            Assert.AreEqual("def", HDFUtils.ExtractString(data, 2, 3));
        }

        [Test]
        public void TestJoin()
        {
            Assert.AreEqual("foo", HDFUtils.Join("/", "foo"));
            Assert.AreEqual("foo", HDFUtils.Join("", "foo"));
            Assert.AreEqual("foo", HDFUtils.Join("foo", "/"));
            Assert.AreEqual("foo", HDFUtils.Join("foo", ""));
            Assert.AreEqual("foo/test", HDFUtils.Join("foo", "test"));
            Assert.AreEqual("/foo/test", HDFUtils.Join("/foo/", "/test"));
            Assert.AreEqual("/foo/test/", HDFUtils.Join("/foo/", "/test/"));
            Assert.AreEqual("foo/test/", HDFUtils.Join("foo/", "/test/"));
            Assert.AreEqual("foo/f2/f3/test", HDFUtils.Join("foo/f2/f3/", "test"));
        }
    }
}