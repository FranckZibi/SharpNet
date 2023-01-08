using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;

namespace SharpNet.Networks
{
    public partial class Network
    {
        #region private fields used by both master and slave networks
        private Tensor _compactedParametersIfAny;
        private Tensor _compactedGradientsIfAny;
        #endregion

        #region private fields used by master network only
        private readonly List<Tensor> all_x_miniBatch = new List<Tensor>();
        /// <summary>
        /// list of all slave networks
        /// empty list of:
        ///     the 'this' network is a slave network
        ///     or
        ///     the 'this' network is a master network  but we are not using multi GPU computation
        ///     (all computation is performed on the master network)
        /// </summary>
        private readonly List<Network> _slaveNetworks = new List<Network>();
        private Tensor _yExpectedForEpoch;
        private Tensor _yPredictedForEpoch;
        #endregion

        #region private fields used by slave networks only
        private SLAVE_NETWORK_STATUS _slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
        /// <summary>
        /// if the current network is a slave network doing computation for its master network:
        ///     the reference of the master network
        /// if current network is a master network:
        ///     null
        /// </summary>
        private readonly Network _masterNetworkIfAny;
        /// <summary>
        /// Tensor x_miniBatch_cpu_slave,
        /// Tensor yExpected_miniBatch_cpu_slave,
        /// Tensor yPredicted_miniBatch_master,
        /// bool isTraining
        /// </summary>
        private Tuple<List<Tensor>, Tensor, Tensor, bool> _slaveParamForMiniBatchGradientDescent;
        private Tensor _yExpected_miniBatch_slave;
        private Tensor _yPredicted_miniBatch_slave;
        #endregion

        public bool IsMaster => _masterNetworkIfAny == null;

        private enum SLAVE_NETWORK_STATUS
        {
            IDLE,
            PREPARE_MINIBATCH_GRADIENT_DESCENT,
            PERFORM_FORWARD_AND_BACKWARD_PROPAGATION,
            TO_ABORT,
            DISPOSED
        }


        #region compaction of parameters and gradients tensors (to speed up communication between master and slaves)
        private void CompactParameters()
        {
            if (_compactedParametersIfAny == null)
            {
                _compactedParametersIfAny = Compact(l => l.Parameters.Select(t => t.Item1).ToList(), (l, tensors) => l.ReplaceParameters(tensors));
            }
        }
        private void CompactGradients()
        {
            if (_compactedGradientsIfAny == null)
            {
                _compactedGradientsIfAny = Compact(l => l.ParameterGradients, (l, tensors) => l.ReplaceGradients(tensors));
            }
        }
        //private void UnCompactParameters()
        //{
        //    if (_compactedParametersIfAny != null)
        //    {
        //        UnCompact(l => l.Parameters.Select(t => t.Item1).ToList(), (l, tensors) => l.ReplaceParameters(tensors));
        //        MemoryPool.FreeFloatTensor(ref _compactedParametersIfAny);
        //    }
        //}
        //private void UnCompactGradients()
        //{
        //    if (_compactedGradientsIfAny != null)
        //    {
        //        UnCompact(l => l.ParameterGradients, (l, tensors) => l.ReplaceGradients(tensors));
        //        MemoryPool.FreeFloatTensor(ref _compactedGradientsIfAny);
        //    }
        //}
        private Tensor Compact(Func<Layer, List<Tensor>> layer2Tensors, Action<Layer, List<Tensor>> storeTensorsInLayer)
        {
            var totalCount = Layers.SelectMany(layer2Tensors).Select(t => t.Count).Sum();
            var compacted = MemoryPool.GetFloatTensor(new[] { totalCount });
            int nextIndex = 0;
            foreach (var l in Layers)
            {
                var layerCompactedTensors = new List<Tensor>();
                foreach (var p in layer2Tensors(l))
                {
                    Debug.Assert(p.IsOwnerOfMemory);
                    var sliceFromCompacted = compacted.Slice(nextIndex, p.Shape);
                    layerCompactedTensors.Add(sliceFromCompacted);
                    p.CopyTo(sliceFromCompacted);
                    nextIndex += sliceFromCompacted.Count;
                }
                storeTensorsInLayer(l, layerCompactedTensors);
            }
            Debug.Assert(nextIndex == totalCount);
            return compacted;
        }
        //private void UnCompact(Func<Layer, List<Tensor>> layer2Tensors, Action<Layer, List<Tensor>> storeTensorsInLayer)
        //{
        //    foreach (var l in Layers)
        //    {
        //        var layerUncompactedParameters = new List<Tensor>();
        //        foreach (var p in layer2Tensors(l))
        //        {
        //            Debug.Assert(!p.IsOwnerOfMemory);
        //            var uncompacted = MemoryPool.GetFloatTensor(p.Shape);
        //            p.CopyTo(uncompacted);
        //            layerUncompactedParameters.Add(uncompacted);
        //        }
        //        storeTensorsInLayer(l, layerUncompactedParameters);
        //    }
        //}
        #endregion

        #region methods used in master network only
        /// <summary>
        /// number of resources (CPU or GPU) that will perform the computation
        /// </summary>
        private void WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS status)
        {
            while (_slaveNetworks.Any(s => s._slaveStatus != status))
            {
                Thread.Sleep(1);
            }
        }
        private void SetStatusForAllSlaves(SLAVE_NETWORK_STATUS newStatus)
        {
            _slaveNetworks.ForEach(s => s._slaveStatus = newStatus);
        }
        private int DegreeOfParallelism
        {
            get
            {
                if (IsMaster)
                {
                    return 1 + _slaveNetworks.Count;
                }
                return _masterNetworkIfAny.DegreeOfParallelism;
            }
        }
        private void AddGradientFromSlaveNetwork(Network slave)
        {
            var master = this;
            Debug.Assert(master.IsMaster);
            Debug.Assert(!slave.IsMaster);
            Debug.Assert(master._compactedGradientsIfAny != null);
            Debug.Assert(slave._compactedGradientsIfAny != null);
            Debug.Assert(master._compactedGradientsIfAny.SameShape(slave._compactedGradientsIfAny));
            var bufferAddGradientFromSlaveNetwork = MemoryPool.GetFloatTensor(master._compactedGradientsIfAny.Shape);
            slave._compactedGradientsIfAny.CopyTo(bufferAddGradientFromSlaveNetwork); //Device to other Device copy (not in the same GPU)
            master._compactedGradientsIfAny.Update_Adding_Alpha_X(1, bufferAddGradientFromSlaveNetwork);
            MemoryPool.FreeFloatTensor(ref bufferAddGradientFromSlaveNetwork);
        }

        #endregion

        #region methods used in slave networks only
        private static void SlaveThread(Network master, bool buildLayers, AbstractDatasetSample datasetSample, int slaveDeviceId)
        {
            //if slave thread will run on a GPU
            if (slaveDeviceId >= 0)
            {
                //we associate the current running (slave) thread with GPU 'slaveDeviceId'
                GPUWrapper.FromDeviceId(slaveDeviceId).AssociateCurrentThreadWithDevice();
            }

            var slaveNetworkSample = (NetworkSample) master.Sample.Clone();
            slaveNetworkSample.ResourceIds = new List<int> { slaveDeviceId };

            var slave = new Network(slaveNetworkSample, datasetSample, master.WorkingDirectory, master.ModelName+"_"+ slaveDeviceId, buildLayers, master);
            lock (master._slaveNetworks)
            {
                master._slaveNetworks.Add(slave);
            }
            slave._spInternalFit.Start();
            for (; ; )
            {
                switch (slave._slaveStatus)
                {
                    case SLAVE_NETWORK_STATUS.PREPARE_MINIBATCH_GRADIENT_DESCENT:
                        if (slave.Layers.Count == 0)
                        {
                            foreach (var l in master.Layers)
                            {
                                l.AddToOtherNetwork(slave);
                            }
                            slave.CompactParameters();
                            slave.CompactGradients();
                        }
                        slave._slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
                        break;
                    case SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION:
                        var param = slave._slaveParamForMiniBatchGradientDescent;
                        if (param == null)
                        {
                            var errorMsg = "null parameters for " + slave._slaveStatus;
                            LogInfo(errorMsg);
                            throw new ArgumentException(errorMsg);
                        }
                        slave.MiniBatchGradientDescentForSlave(param.Item1, param.Item2, param.Item3, param.Item4);
                        slave._slaveParamForMiniBatchGradientDescent = null;
                        slave._slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
                        break;
                    case SLAVE_NETWORK_STATUS.TO_ABORT:
                        slave._spInternalFit.Stop();
                        LogDebug("stopping thread for network " + slave.ModelName);
                        LogDebug(slave.MemoryInfo());
                        LogDebug(slave.LayersKpi());
                        slave.Dispose();
                        slave._slaveStatus = SLAVE_NETWORK_STATUS.DISPOSED;
                        return;
                    default:
                        //case SLAVE_NETWORK_STATUS.IDLE:
                        Thread.Sleep(1);
                        break;
                }
            }
        }
        private void MiniBatchGradientDescentForSlave(List<Tensor> all_x_miniBatch_cpu_slave, Tensor yExpected_miniBatch_cpu_slave, Tensor yPredicted_miniBatch_master, bool isTraining)
        {
            Debug.Assert(_yPredictedForEpoch == null);
            Debug.Assert(_yExpectedForEpoch == null);
            Debug.Assert(yExpected_miniBatch_cpu_slave.SameShape(yPredicted_miniBatch_master));
            Debug.Assert(all_x_miniBatch_cpu_slave[0].Shape[0] == yExpected_miniBatch_cpu_slave.Shape[0]);
            Debug.Assert(!all_x_miniBatch_cpu_slave[0].UseGPU);
            Debug.Assert(!yExpected_miniBatch_cpu_slave.UseGPU);
            Debug.Assert(yPredicted_miniBatch_master.UseGPU == UseGPU);
            Debug.Assert(_masterNetworkIfAny._compactedParametersIfAny != null);
            Debug.Assert(_compactedParametersIfAny != null);

            //TODO try to do this copy in the master network and not in the slave network
            //We copy the weights from the master network to the slave network
            StartTimer("CopyWeights_Master2Slave", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
            _masterNetworkIfAny._compactedParametersIfAny.CopyTo(_compactedParametersIfAny);
            StopTimer("CopyWeights_Master2Slave", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

            //we initialize '_xMiniBatch' & '_yExpected_miniBatch_slave'
            for (int x = 0; x < all_x_miniBatch_cpu_slave.Count; ++x)
            {
                if (all_x_miniBatch.Count <= x)
                {
                    all_x_miniBatch.Add(MemoryPool.GetFloatTensor(all_x_miniBatch_cpu_slave[x].Shape));
                }
                else
                {
                    var tmp_x_miniBatch = all_x_miniBatch[x];
                    MemoryPool.GetFloatTensor(ref tmp_x_miniBatch, all_x_miniBatch_cpu_slave[x].Shape);
                }
            }

           

            MemoryPool.GetFloatTensor(ref _yExpected_miniBatch_slave, yExpected_miniBatch_cpu_slave.Shape);
            yExpected_miniBatch_cpu_slave.CopyTo(_yExpected_miniBatch_slave);
            MemoryPool.GetFloatTensor(ref _yPredicted_miniBatch_slave, _yExpected_miniBatch_slave.Shape);
            PropagationManager.Forward(all_x_miniBatch, _yPredicted_miniBatch_slave, isTraining);
            if (isTraining)
            {
                PropagationManager.Backward(_yExpected_miniBatch_slave, _yPredicted_miniBatch_slave, Sample.LossFunction);
            }

            //copy miniBatch prediction (computed in slave network) to master network
            _yPredicted_miniBatch_slave.CopyTo(yPredicted_miniBatch_master);
        }
        #endregion
    }
}
