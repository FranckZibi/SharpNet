using System;
using System.Linq;
using SharpNet.Layers;

namespace SharpNet.Networks
{
    public partial class Network
    {
        /// <summary>
        /// set the number of output categories of the current network by updating the head layers (Dense+Activation layers)
        /// if the number of output categories is already 'newCategoryCount'
        ///     does nothing at all
        /// else
        ///     update the last Dense Layers (resetting all its weights) to match the required number of categories
        /// </summary>
        /// <param name="newCategoryCount">the target number of categories</param>
        public void SetCategoryCount(int newCategoryCount)
        {
            LogInfo("setting number of output categoryCount to " + newCategoryCount);
            if (Layers.Count >= 2 && Layers[Layers.Count - 1] is ActivationLayer && Layers[Layers.Count - 2] is DenseLayer)
            {
                var denseLayer = (DenseLayer) Layers[Layers.Count - 2];
                if (denseLayer.Units == newCategoryCount)
                {
                    LogInfo("no need to set the CategoryCount to " + newCategoryCount);
                    return; //already at target category count
                }

                //we remove the ActivationLayer (last layer)
                var activationLayer = (ActivationLayer) Layers.Last();
                var activationFunctionType = activationLayer.ActivationFunction;
                var activationLayerName = activationLayer.LayerName;
                RemoveAndDisposeLastLayer();

                //we remove the Dense layer
                var lambdaL2Regularization = denseLayer.LambdaL2Regularization;
                var denseLayerName = denseLayer.LayerName;
                RemoveAndDisposeLastLayer();

                //We add a new DenseLayer (with weight reseted)
                LogInfo("Resetting weights of layer " + denseLayerName + " to have " + newCategoryCount + " categories");
                Dense(newCategoryCount, lambdaL2Regularization, false, denseLayerName);

                //we put back the ActivationLayer
                Activation(activationFunctionType, activationLayerName);

                return;
            }

            throw new NotImplementedException(
                "can only update a network where the 2 last layers are DenseLayer & ActivationLayer");
        }

        private void RemoveAndDisposeLastLayer()
        {
            var lastLayer = Layers.Last();
            foreach (var previousLayers in lastLayer.PreviousLayers)
            {
                previousLayers.NextLayerIndexes.Remove(lastLayer.LayerIndex);
            }

            lastLayer.Dispose();
            Layers.RemoveAt(Layers.Count - 1);
            OnLayerAddOrRemove();
        }

        public void FreezeSelectedLayers()
        {
            if (string.IsNullOrEmpty(Config.FirstLayerNameToFreeze) &&
                string.IsNullOrEmpty(Config.LastLayerNameToFreeze))
            {
                //no layers to freeze : we'll train the entire network
                return;
            }

            var firstLayerIndexToFreeze = LayerNameToLayerIndex(Config.FirstLayerNameToFreeze);
            if (firstLayerIndexToFreeze == -1)
            {
                firstLayerIndexToFreeze = 0;
            }

            var lastLayerIndexToFreeze = LayerNameToLayerIndex(Config.LastLayerNameToFreeze);
            if (lastLayerIndexToFreeze == -1)
            {
                lastLayerIndexToFreeze = LastLayerIndex;
            }

            LogInfo("Freezing " + (lastLayerIndexToFreeze - firstLayerIndexToFreeze + 1) + " layers (between " +
                     Layers[firstLayerIndexToFreeze].LayerName + " and " + Layers[lastLayerIndexToFreeze].LayerName + ")");
            for (int layerIndex = 0; layerIndex < Layers.Count; ++layerIndex)
            {
                var layer = Layers[layerIndex];
                if (layerIndex >= firstLayerIndexToFreeze && layerIndex <= lastLayerIndexToFreeze)
                {
                    //we need to freeze the weights/bias associated with the layer
                    layer.Trainable = false;
                }
                else
                {
                    //the layer is trainable
                    layer.Trainable = true;
                    //we reset the layer weights to their default values
                    layer.ResetParameters();
                }
            }
        }

        /// <summary>
        /// return the index of the layer whose name is 'layerName' or -1 if there is no such layer
        /// </summary>
        /// <param name="layerName">the layer name for which we want to know the layer index</param>
        /// <returns></returns>
        private int LayerNameToLayerIndex(string layerName)
        {
            for (var layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
            {
                var layer = Layers[layerIndex];
                if (string.Equals(layer.LayerName, layerName ?? "", StringComparison.OrdinalIgnoreCase))
                {
                    return layerIndex;
                }
            }

            return -1;
        }
    }
}