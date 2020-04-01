using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;

namespace SharpNetTests
{
    [TestFixture]
    public class TestCop
    {
        /// <summary>
        /// The goal of the method is to ensure that the source code follow some rules
        /// </summary>
        [Test]
        public void TestSourceCode()
        {
            //the list of rules the src code must follow
            var srcCodeRules = new List<Func<string, string>>
            {
                CheckMixTensorCountTensorCapacity
            };
            var warnings = new List<string>();
            foreach (var filePath in AllSourceFilePath("Tests"))
            {
                var lines = File.ReadAllLines(filePath);
                for (var lineIndex = 0; lineIndex < lines.Length; lineIndex++)
                {
                    var line = lines[lineIndex];
                    foreach (var check in srcCodeRules)
                    {
                        var checkResult = check(line);
                        if (!string.IsNullOrEmpty(checkResult))
                        {
                            warnings.Add("file " + filePath + " line " + (lineIndex + 1) + " : " + checkResult + Environment.NewLine + line);
                        }
                    }
                }
            }
            Assert.True(warnings.Count == 0, string.Join(Environment.NewLine, warnings));
        }

        /// <summary>
        /// retrieve the C# source code files in the current project
        /// this source code is at the same level as the provided 'srcCodeDirectory' directory name 
        /// </summary>
        /// <param name="srcCodeDirectory">name of a directory.
        /// The src code is at the same level of this directory</param>
        /// <returns></returns>
        private static IEnumerable<string> AllSourceFilePath(string srcCodeDirectory)
        {
            var path = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            int indexOfSrcCodDirectory = path.LastIndexOf(srcCodeDirectory, StringComparison.OrdinalIgnoreCase);
            if (indexOfSrcCodDirectory < 0)
            {
                return Enumerable.Empty<string>();
            }
            return Directory.GetFiles(path.Substring(0, indexOfSrcCodDirectory), "*.cs", SearchOption.AllDirectories);
        }

        #region list of source code rules
        /// <summary>
        /// check if the source code line provided as input may have a problem because it can mixes
        /// the tensor count and the tensor capacity
        /// </summary>
        /// <param name="sourceCodeLine">the source line to check</param>
        /// <returns>
        /// empty string if no problem detected
        /// a warning msg if a problem has been detected
        /// </returns>
        private static string CheckMixTensorCountTensorCapacity(string sourceCodeLine)
        {
            if (sourceCodeLine.Contains("Content" + ".Length", StringComparison.OrdinalIgnoreCase)
                && !sourceCodeLine.Contains("CapacityInBytes", StringComparison.OrdinalIgnoreCase)
            )
            {
                return "possible mix between Tensor length and Tensor capacity";
            }
            return ""; //no problem
        }
        #endregion

    }
}