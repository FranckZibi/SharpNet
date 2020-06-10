using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCancelDatabase
    {
        [TestCase(new string[]{}, "")]
        [TestCase(new[]{ "used", "gc" }, "gc")]
        [TestCase(new[]{ "used", "gc", "1digit" }, "gc*")]
        [TestCase(new[]{ "used", "gc", "1digit", "7" }, "gc7")]
        [TestCase(new[] { "used", "gc", "2digits" }, "gc**")]
        [TestCase(new[] { "used", "gc", "2digits", "*" , "7"}, "gc*7")]
        [TestCase(new[] { "used", "gc", "3digits", "2", "*" , "7"}, "gc2*7")]
        [TestCase(new[] { "used", "gc", "4digits", "4", "*" , "7", "*"}, "gc4*7*")]
        [TestCase(new[] { "used", "gc", "4digits", "4", "0", "7", "9" }, "gc4079")]
        [TestCase(new[] { "used", "star", "1digit"}, "etoile*")]
        [TestCase(new[] { "used", "star", "1digit", "7"}, "etoile7")]
        [TestCase(new[] { "used", "star", "2digits"}, "etoile**")]
        [TestCase(new[] { "used", "star", "2digits", "1", "*"}, "etoile1*")]
        [TestCase(new[] { "used", "star", "2digits", "*", "7"}, "etoile*7")]
        [TestCase(new[] { "used", "star", "2digits", "1", "7"}, "etoile17")]
        public void TestToPath(string[] expectedPath, string cancelName)
        {
            Assert.AreEqual(expectedPath, CancelDatabase.ToPath(cancelName));
        }

        [Test]
        public void TestCategoryNameToPrediction()
        {
            var available = new List<char> {'1','2','3','*'};
            for (int length = 1; length <= 4; ++length)
            {
                foreach(var expectedCancelName in AllCombinations(available, length))
                {
                    TestCategoryNameToPrediction("gc"+expectedCancelName);
                    TestCategoryNameToPrediction("pc"+expectedCancelName);
                    if (length <= 2)
                    {
                        TestCategoryNameToPrediction("etoile" + expectedCancelName);
                    }
                }
            }
        }


        private static void TestCategoryNameToPrediction(string expectedCancelName)
        {
            var path = CancelDatabase.ToPath(expectedCancelName);
            var prediction = CancelDatabase.Hierarchy.ExpectedPrediction(path);
            var observedCancelName = CancelDatabase.Hierarchy.ExtractPrediction(prediction);
            Assert.AreEqual(expectedCancelName, observedCancelName);
        }

        private static List<string> AllCombinations(List<char> available, int length)
        {
            var allItems = new List<List<char>>();
            while (allItems.Count < length)
            {
                allItems.Add(available);
            }
            var result = new List<string>();
            foreach (var comb in AllCombinations(allItems))
            {
                result.Add(new string(comb.ToArray()));
            }
            return result;
        }

        private static IEnumerable<List<T>> AllCombinations<T>(IList<List<T>> allItems)
        {
            var result = new List<List<T>>();
            if ((allItems == null) || (allItems.Count == 0) || allItems.Any(x => (x == null) || (x.Count == 0)))
            {
                return result;
            }
            AllCombinationsHelper(allItems, new List<T>(), result);
            return result;
        }
        private static void AllCombinationsHelper<T>(IList<List<T>> allItems, List<T> currentSolutionInProgress, List<List<T>> allSolutionSoFar)
        {
            if (currentSolutionInProgress.Count == allItems.Count)
            {
                allSolutionSoFar.Add(currentSolutionInProgress); // a new combination has been found
                return;
            }
            foreach (var t in allItems[currentSolutionInProgress.Count])
            {
                AllCombinationsHelper(allItems, new List<T>(currentSolutionInProgress) { t }, allSolutionSoFar);
            }
        }

    }
}