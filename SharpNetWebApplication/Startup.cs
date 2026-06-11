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
        private static Network Network;
        //temporary files (one per picture content received through the REST API) waiting to be processed
        private static readonly List<string> ToProcess =  new ();
        //directory where the picture contents received through the REST API are stored while waiting to be processed
        private static readonly string TempPictureDirectory = InitializeTempPictureDirectory();
        #endregion
        #region public fields
        private static readonly IDictionary<string, CancelIdentification> Cache = new ConcurrentDictionary<string, CancelIdentification>();
        #endregion

        private static void ComputationThread()
        {
            Network = CancelDatabase.GetDefaultNetwork();

            Log.Info("ComputationThread is starting");
            Log.Info(Network.Summary());
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
                    var predictions = CancelDatabase.PredictCancelsWithProba(Network, picturePaths);
                    for (var index = 0; index < predictions.Count; index++)
                    {
                        //the temporary picture file is named '<id>.img' (with 'id' the hash of the picture content)
                        var id = Path.GetFileNameWithoutExtension(picturePaths[index]);
                        if (!Cache.TryGetValue(id, out var cancelIdentification))
                        {
                            cancelIdentification = new CancelIdentification {StartComputationDate = DateTime.Now};
                        }
                        cancelIdentification.Id = id;
                        cancelIdentification.IsDone = true;
                        cancelIdentification.Prediction = predictions[index].Item1;
                        cancelIdentification.PredictionProbability = predictions[index].Item2;
                        cancelIdentification.ComputationTimeInMilliseconds = (int)(DateTime.Now - cancelIdentification.StartComputationDate).TotalMilliseconds;
                        Program.TotalMilliSecondsForAllRequests += cancelIdentification.ComputationTimeInMilliseconds;
                        Cache[id] = cancelIdentification;
                        Log.Debug("Identification of "+ cancelIdentification .Id+ " : "+ cancelIdentification.Prediction + " (proba:"+Math.Round(100*cancelIdentification.PredictionProbability,1)+"%)");
                    }
                    foreach (var picturePath in picturePaths)
                    {
                        TryDeleteTempPictureFile(picturePath);
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


        /// <summary>
        /// Starts the cancel identification of the provided picture content (if it is not already available).
        /// </summary>
        /// <param name="pictureContent">The raw bytes of the picture to identify.</param>
        /// <returns>
        /// The (possibly not yet done) cancel identification associated with the picture content.
        /// Its <see cref="CancelIdentification.Id"/> is the hash of the content, to be used to retrieve the result later.
        /// </returns>
        /// <remarks>
        /// Not unit tested: this method relies on the static computation thread (which loads a trained neural network)
        /// and on disk I/O. The pure id computation is tested behind <see cref="CancelIdentificationServices.ComputeId"/>.
        /// </remarks>
        public static CancelIdentification AddComputation(byte[] pictureContent)
        {
            if (pictureContent == null || pictureContent.Length == 0)
            {
                Log.Warn("the provided picture content is empty");
                return new CancelIdentification { Id = "", IsDone = true, Prediction = "Invalid content", PredictionProbability = 0.0, StartComputationDate = DateTime.Now, ComputationTimeInMilliseconds = 0};
            }
            var id = CancelIdentificationServices.ComputeId(pictureContent);
            Log.Debug("AddComputation of picture content with id "+id);
            Interlocked.Increment(ref Program.NbRequest);
            if (Cache.ContainsKey(id))
            {
                return Cache[id];
            }
            Cache[id] = new CancelIdentification { Id = id, IsDone = false, Prediction = "", PredictionProbability = 0.0, StartComputationDate = DateTime.Now, ComputationTimeInMilliseconds = 0 };
            var picturePath = Path.Combine(TempPictureDirectory, id + ".img");
            File.WriteAllBytes(picturePath, pictureContent);
            Log.Debug("Add "+picturePath+" to the process list");

            lock (ToProcess)
            {
                ToProcess.Add(picturePath);
            }
            return Cache[id];
        }


        /// <summary>
        /// Gets the cancel identification associated with the provided id (the hash of the picture content).
        /// </summary>
        /// <param name="id">The id returned by <see cref="AddComputation"/>.</param>
        /// <returns>
        /// The cancel identification if the id is known (its <see cref="CancelIdentification.IsDone"/> flag tells
        /// if the prediction is available), or a 'not done' result for an unknown id.
        /// </returns>
        /// <remarks>
        /// Not unit tested: this method relies on the static cache filled by the computation thread
        /// (which loads a trained neural network and cannot be exercised deterministically in a unit test).
        /// </remarks>
        public static CancelIdentification GetCancelIdentificationIfAvailable(string id)
        {
            Log.Debug("GetCancelIdentificationIfAvailable of " + id);
            if (Cache.TryGetValue(id, out var cancelIdentification))
            {
                return cancelIdentification;
            }
            Log.Warn("unknown cancel identification id " + id);
            return new CancelIdentification { Id = id, IsDone = false, Prediction = "Unknown id", PredictionProbability = 0.0, StartComputationDate = DateTime.Now, ComputationTimeInMilliseconds = 0 };
        }

        private static string InitializeTempPictureDirectory()
        {
            var tempPictureDirectory = Path.Combine(Path.GetTempPath(), "SharpNetWebApplication");
            Directory.CreateDirectory(tempPictureDirectory);
            return tempPictureDirectory;
        }

        private static void TryDeleteTempPictureFile(string picturePath)
        {
            try
            {
                File.Delete(picturePath);
            }
            catch (Exception e)
            {
                Log.Warn("Fail to delete the temporary picture file " + picturePath, e);
            }
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
