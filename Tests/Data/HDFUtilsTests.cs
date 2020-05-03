using System;
using HDF.PInvoke;
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

        [TestCase("/file", "/", "file")]
        [TestCase("file", "", "file")]
        [TestCase("/toto/toto2/file", "/toto/toto2/", "file")]
        [TestCase("toto/toto2/file", "toto/toto2/", "file")]
        [TestCase("/toto/toto2/file/", "/toto/toto2/file/", "")]
        [TestCase("/toto/", "/toto/", "")]
        [TestCase("/toto/file", "/toto/", "file")]
        public void TestSplitPathAndName(string datasetPathWithName, string expectedPath, string expectedName)
        {
            HDFUtils.SplitPathAndName(datasetPathWithName, out string path, out string name);
            Assert.AreEqual(expectedPath, path);
            Assert.AreEqual(expectedName, name);
        }

        [Test]
        public void TestToH5TypeId()
        {
            Assert.AreEqual(H5T.NATIVE_FLOAT, HDFUtils.ToH5TypeId(typeof(float)));
            Assert.Throws<NotImplementedException>(()=> HDFUtils.ToH5TypeId(typeof(string)));
        }

        [Test]
        public void TestJoin()
        {
            Assert.AreEqual("/foo", HDFUtils.Join("/", "foo"));
            Assert.AreEqual("/foo/test", HDFUtils.Join("/", "foo/test"));
            Assert.AreEqual("/foo", HDFUtils.Join("", "foo"));
            Assert.AreEqual("foo/", HDFUtils.Join("foo", "/"));
            Assert.AreEqual("foo", HDFUtils.Join("foo", ""));
            Assert.AreEqual("foo/test", HDFUtils.Join("foo", "test"));
            Assert.AreEqual("/foo/test", HDFUtils.Join("/foo/", "/test"));
            Assert.AreEqual("/foo/test/", HDFUtils.Join("/foo/", "/test/"));
            Assert.AreEqual("foo/test/", HDFUtils.Join("foo/", "/test/"));
            Assert.AreEqual("foo/f2/f3/test", HDFUtils.Join("foo/f2/f3/", "test"));
        }
    }
}
