using System;
using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet
{
    public class Logger
    {
        #region fields
        private readonly string _logFileName;
        private readonly bool _logInConsole;
        public static readonly Logger ConsoleLogger = new Logger("", true);
        public static readonly Logger NullLogger = new Logger("", false);
        #endregion

        public Logger(string logFileName, bool logInConsole)
        {
            _logFileName = logFileName ?? "";
            _logInConsole = logInConsole;
        }
        public void Info(string msg)
        {
            if (_logInConsole)
            {
                Console.WriteLine(msg);
            }
            LogInFile(msg);
        }
        public void Debug(string msg)
        {
            LogInFile(msg);
        }
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_logFileName), _logFileName)
                .Add(nameof(_logInConsole), _logInConsole)
                .ToString();
        }
        public static Logger ValueOf(IDictionary<string, object> serialized)
        {
            var logFileName = (string)serialized[nameof(_logFileName)];
            var logInConsole = (bool)serialized[nameof(_logInConsole)];
            return new Logger(logFileName, logInConsole);
        }

        private static string GetLinePrefix()
        {
            return DateTime.Now.ToString("HH:mm:ss.ff") + " ";
        }
        private void LogInFile(string msg)
        {
            if (string.IsNullOrEmpty(_logFileName))
            {
                return;
            }
            lock (_logFileName)
            {
                Utils.AddLineToFile(_logFileName, GetLinePrefix() + msg);
            }
        }
    }
}