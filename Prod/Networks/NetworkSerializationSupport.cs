using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;

namespace SharpNet.Networks
{
    public partial class Network
    {

        /// <summary>
        /// ensure that the resource ids required for a network are available, and fix them if it is not
        /// </summary>
        /// <param name="resourceIds">the requested resource ids for the network</param>
        /// <param name="gpuCount">number of GPUs available in the current computer</param>
        /// <returns>the fixed list of resource ids for the network</returns>
        public static int[] AdaptResourceIdsToCurrentComputer(int[] resourceIds, int gpuCount)
        {
            var fixedResult = new List<int>();
            foreach (var id in resourceIds)
            {
                if (id < gpuCount)
                {
                    fixedResult.Add(id);
                }
            }
            if (fixedResult.Count == 0)
            {
                fixedResult.Add(-1);
            }
            return fixedResult.ToArray();
        }
        
        public static Network LoadTrainedNetworkModel(string workingDirectory, string modelName)
        {
            var modelFilePath = Path.Combine(workingDirectory, modelName + ".txt");
            //we load the model (network description)
            var allLines = File.ReadAllLines(modelFilePath);
            var dicoFirstLine = Serializer.Deserialize(allLines[0]);
            var sample = NetworkSample.ValueOf(workingDirectory, modelName);
            if (!string.IsNullOrEmpty(sample.Config.WorkingDirectory) && !Directory.Exists(sample.Config.WorkingDirectory))
            {
                // ReSharper disable once PossibleNullReferenceException
                sample.Config.WorkingDirectory = new FileInfo(modelFilePath).Directory.FullName;
            }

            var originalResourceIds = sample.Config.ResourceIds.ToArray();
            int[] fixedResourceIds = AdaptResourceIdsToCurrentComputer(originalResourceIds, GPUWrapper.GetDeviceCount());
            sample.Config.ResourceIds = fixedResourceIds.ToList();

            var network = new Network(sample);
            if (!originalResourceIds.SequenceEqual(fixedResourceIds))
            {
                LogWarn("changing resourceIds from ("+string.Join(",", originalResourceIds)+ ") to ("+string.Join(",", fixedResourceIds)+")");
            }
            //on CPU we must use 'GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM'
            if (fixedResourceIds.Max() < 0 && sample.Config.ConvolutionAlgoPreference != GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                LogWarn("only " + GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM + " is available on CPU (" + sample.Config.ConvolutionAlgoPreference + " is not supported on CPU)");
                sample.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
                LogWarn("force using " + GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM);
            }

            var epochsData = (EpochData[])dicoFirstLine[nameof(EpochData)];
            network.EpochData.AddRange(epochsData);
            network.Description = dicoFirstLine.TryGet<string>(nameof(Description)) ?? "";
            for (int i = 1; i < allLines.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(allLines[i]), network));
            }

            //we load the parameters into the network
            var parametersFilePath = ToParameterFilePath(workingDirectory, modelName);
            if (File.Exists(parametersFilePath))
            {
                network.LoadParametersFromH5File(parametersFilePath, network.Config.CompatibilityMode);
            }
            return network;
        }

        /// <summary>
        /// save network the parameters in h5 file 'parametersFilePath'
        /// if it already exist, it will be removed first
        /// </summary>
        private void SaveParameters(string workingDirectory, string modelName)
        {
            var swSaveParametersTime = Stopwatch.StartNew();

            //the 5h file where to store the network parameters.
            var parametersFilePath = ToParameterFilePath(workingDirectory, modelName);
            if (File.Exists(parametersFilePath))
            {
                File.Delete(parametersFilePath);
            }
            using var h5File = new H5File(parametersFilePath);
            foreach (var l in Layers)
            {
                foreach (var p in l.GetParametersAsCpuFloatTensors(Config.CompatibilityMode))
                {
                    h5File.Write(p.Key, p.Value);
                }
            }
            LogInfo("Network Parameters '" + Description + "' saved in " + parametersFilePath + " in " + Math.Round(swSaveParametersTime.Elapsed.TotalSeconds, 1) + "s");
        }

        /// <summary>
        /// load the parameters from h5 file 'h5FilePath' into the network
        /// </summary>
        /// <param name="h5FilePath"></param>
        /// <param name="originFramework"></param>
        public void LoadParametersFromH5File(string h5FilePath, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            LogInfo("loading weights from " + h5FilePath);
            using var h5File = new H5File(h5FilePath);
            var h5FileParameters = h5File.Datasets();
            Layers.ForEach(l => l.LoadParameters(h5FileParameters, originFramework));

            var networkParametersKeys = Layers.SelectMany(t => t.Parameters).Select(t => t.Item2).ToList();
            var elementMissingInH5Files = networkParametersKeys.Except(h5FileParameters.Keys).ToList();
            if (elementMissingInH5Files.Count != 0)
            {
                LogInfo(elementMissingInH5Files.Count + " parameters are missing in file " + h5FilePath + ": " +
                         string.Join(", ", elementMissingInH5Files));
            }

            var elementMissingInNetwork = h5FileParameters.Keys.Except(networkParametersKeys).ToList();
            if (elementMissingInNetwork.Count != 0)
            {
                LogInfo(elementMissingInNetwork.Count + " parameters are missing in network " + h5FilePath + ": " +
                         string.Join(", ", elementMissingInNetwork));
            }
        }


        public override List<string> ModelFiles()
        {
            return new List<string>
            {
                ToModelFilePath(WorkingDirectory, ModelName),
                ToParameterFilePath(WorkingDirectory, ModelName)
            };
        }

        [NotNull] public static string ToModelFilePath(string workingDirectory, string modelName)
        {
            return Path.Combine(workingDirectory, modelName + ".txt");
        }
        public static string ToParameterFilePath(string workingDirectory, string modelName)
        {
            return Path.Combine(workingDirectory, modelName + ".h5");
        }

        /// <summary>
        /// save network model
        /// if it already exist, it will be removed first
        /// </summary>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName"></param>
        private void SaveModel(string workingDirectory, string modelName)
        {
            var swSaveModelTime = Stopwatch.StartNew();

            var modelFilePath = ToModelFilePath(workingDirectory, modelName);
            if (File.Exists(modelFilePath))
            {
                File.Delete(modelFilePath);
            }
            ModelSample.Save(workingDirectory, modelName);


            var firstLine = new Serializer()
                .Add(nameof(Description), Description)
                .Add(nameof(EpochData), EpochData.ToArray())
                .ToString();
            File.AppendAllLines(modelFilePath, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(modelFilePath, new[] { l.Serialize() });
            }

            LogInfo("Network Model '" + Description + "' saved in " + modelFilePath + " in " + Math.Round(swSaveModelTime.Elapsed.TotalSeconds, 1) + "s");
        }

        public override void Save(string workingDirectory, string sampleName)
        {
            SaveModel(workingDirectory, sampleName);
            SaveParameters(workingDirectory, sampleName);
        }
    }
}
