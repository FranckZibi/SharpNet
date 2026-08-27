using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using log4net;
using Path = System.IO.Path;

namespace SharpNet
{
    public static partial class Utils
    {
        private static int? _cacheCoreCount;
        public static int CoreCount
        {
            get
            {
                if (!_cacheCoreCount.HasValue)
                {
                    int coreCount = 0;
                    foreach (var item in new System.Management.ManagementObjectSearcher("Select * from Win32_Processor").Get())
                    {
                        coreCount += int.Parse(item["NumberOfCores"].ToString() ?? "");
                    }
                    _cacheCoreCount = coreCount;
                }

                return _cacheCoreCount.Value;

            }
        }
        public static List<string> Launch(string workingDirectory, string exePath, string arguments, ILog log, bool returnOutputedLines)
        {
            var outputLines = returnOutputedLines?new List<string>():null;
            Log.Debug($"Launching {exePath} {arguments} with WorkingDirectory={workingDirectory}");
            var errorDataReceived = "";
            var engineName = Path.GetFileNameWithoutExtension(exePath);
            var psi = new ProcessStartInfo(exePath)
            {
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                Arguments = arguments,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };
            var process = Process.Start(psi);
            if (process == null)
            {
                string errorMsg = "Fail to start " + engineName + " Engine";
                log.Fatal(errorMsg);
                throw new Exception(errorMsg);
            }
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    errorDataReceived = e.Data;
                }
            };
            process.OutputDataReceived += (_, e) =>
            {
                outputLines?.Add(e.Data);
                if (string.IsNullOrEmpty(e.Data)
                    || e.Data.Contains("Object info sizes") 
                    || e.Data.Contains("Skipping test eval output") 
                    || e.Data.Contains(" min passed")
                    || e.Data.Contains("No further splits with positive gain")
                    || e.Data.Contains("remaining:")
                    || e.Data.Contains("seconds elapsed")
                    || e.Data.Contains("[Info] Iteration:")
                   )
                {
                    return;
                }
                log.Debug(e.Data);
            };
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            process.WaitForExit();
            if (!string.IsNullOrEmpty(errorDataReceived) || process.ExitCode != 0)
            {
                if (!(errorDataReceived??"").Contains("is not implemented on GPU"))
                {
                    var errorMsg = "Error in " + engineName + " " + errorDataReceived;
                    log.Fatal(errorMsg);
                    throw new Exception(errorMsg);
                }
            }
            return outputLines;
        }
        /// <summary>
        /// process the log of a model to look for values after some specific token
        /// the last value found for a token is always the one to use
        /// </summary>
        /// <param name="lines"></param>
        /// <param name="indexValueAfterToken"></param>
        /// <param name="tokenAndMandatoryItemAfterToken"></param>
        /// <returns></returns>
        public static double[] ExtractValuesFromOutputLog(IEnumerable<string> lines, int indexValueAfterToken, params string[] tokenAndMandatoryItemAfterToken)
        {
            Debug.Assert(tokenAndMandatoryItemAfterToken.Length%2 == 0);
            var token = new string[tokenAndMandatoryItemAfterToken.Length / 2];
            var mandatoryItemAfterToken = new string[token.Length];
            for (int i = 0; i < tokenAndMandatoryItemAfterToken.Length; i += 2)
            {
                token[i / 2] = tokenAndMandatoryItemAfterToken[i];
                mandatoryItemAfterToken[i / 2] = tokenAndMandatoryItemAfterToken[i + 1];
            }

            var results = Enumerable.Repeat(double.NaN, token.Length).ToArray();
            foreach(var line in lines.Reverse())
            {
                if (string.IsNullOrEmpty(line))
                {
                    continue;
                }
                if (results.All(val => !double.IsNaN(val)))
                {
                    return results; //we already have filled all values, no need to look in other lines
                }
                for (var j = 0; j < token.Length; j++)
                {
                    if (!double.IsNaN(results[j]))
                    {
                        continue; //we have already filled the value for token 'token[j]'
                    }
                    int idx = line.IndexOf(token[j], StringComparison.Ordinal);
                    if (idx < 0)
                    {
                        continue;
                    }
                    var splitted = line.Substring(idx + token[j].Length).Trim().Split();
                    if (   indexValueAfterToken< splitted.Length
                           && (mandatoryItemAfterToken[j] == null || mandatoryItemAfterToken[j] == splitted[0])
                           && double.TryParse(splitted[indexValueAfterToken], out var d))
                    {
                        results[j] = d;
                    }
                }
            }
            return results;
        }
    }
}
