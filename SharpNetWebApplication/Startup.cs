using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using log4net;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SharpNet.Datasets;
using SharpNet.Networks;
using SharpNetWebApplication.Models;

namespace SharpNetWebApplication
{
    public class Startup
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Startup));
        
        #region private fields
        private static Network _network;
        private static readonly List<string> ToProcess =  new ();
        #endregion
        #region public fields
        private static readonly IDictionary<string, CancelIdentification> Cache = new ConcurrentDictionary<string, CancelIdentification>();
        #endregion

        private static void ComputationThread()
        {
            _network = CancelDatabase.GetDefaultNetwork();

            Log.Info("ComputationThread is starting");
            Log.Info(_network.Summary());
            for (;;)
            {
                var picturePaths = new List<string>();
                lock (ToProcess)
                {
                    picturePaths.AddRange(ToProcess);
                    ToProcess.Clear();
                }
                if (picturePaths.Count != 0)
                {
                    var predictions = CancelDatabase.PredictCancelsWithProba(_network, picturePaths);
                    for (var index = 0; index < predictions.Count; index++)
                    {
                        if (!Cache.TryGetValue(picturePaths[index], out var cancelIdentification))
                        {
                            cancelIdentification = new CancelIdentification {StartComputationDate = DateTime.Now};
                        }
                        cancelIdentification.Id = picturePaths[index];
                        cancelIdentification.IsDone = true;
                        cancelIdentification.Prediction = predictions[index].Item1;
                        cancelIdentification.PredictionProbability = predictions[index].Item2;
                        cancelIdentification.ComputationTimeInMilliseconds = (int)(DateTime.Now - cancelIdentification.StartComputationDate).TotalMilliseconds;
                        Program.TotalMilliSecondsForAllRequests += cancelIdentification.ComputationTimeInMilliseconds;
                        Cache[picturePaths[index]] = cancelIdentification;
                        Log.Debug("Identification of "+ cancelIdentification .Id+ " : "+ cancelIdentification.Prediction + " (proba:"+Math.Round(100*cancelIdentification.PredictionProbability,1)+"%)");
                    }
                }
                if (Cache.Count > 1000)
                {
                    //we remove old entries (+30 minuts)
                    foreach (var key in Cache.Keys.ToList())
                    {
                        if (Cache.TryGetValue(key, out var c) && (DateTime.Now - c.StartComputationDate).TotalMinutes > 30)
                        {
                            Cache.Remove(key);
                        }
                    }
                }
                Thread.Sleep(1);
            }
            // ReSharper disable once FunctionNeverReturns
        }


        public static CancelIdentification AddComputation(string path)
        {
            Log.Debug("AddComputation of "+path);

            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Log.Warn(path+" is invalid or missing");
                return new CancelIdentification { Id = path, IsDone = true, Prediction = "Invalid path", PredictionProbability = 0.0, StartComputationDate = DateTime.Now, ComputationTimeInMilliseconds = 0};
            }
            Interlocked.Increment(ref Program.NbRequest);
            if (Cache.ContainsKey(path))
            {
                return Cache[path];
            }
            Cache[path] = new CancelIdentification { Id = path, IsDone = false, Prediction = "", PredictionProbability = 0.0, StartComputationDate = DateTime.Now, ComputationTimeInMilliseconds = 0 };
            Log.Debug("Add "+path+" to the process list");

            lock (ToProcess)
            {
                ToProcess.Add(path);
            }
            return Cache[path];
        }


        public static CancelIdentification GetCancelIdentificationIfAvailable(string path)
        {
            Log.Debug("GetCancelIdentificationIfAvailable of " + path);
            return Cache.ContainsKey(path) ? Cache[path] : AddComputation(path);
        }

        static Startup()
        {
            new Thread(ComputationThread).Start();
        }

        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once UnusedAutoPropertyAccessor.Global
        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers();
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        // ReSharper disable once UnusedMember.Global
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.UseHttpsRedirection();

            app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}
