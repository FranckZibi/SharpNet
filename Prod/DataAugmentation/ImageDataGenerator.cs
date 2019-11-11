using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class ImageDataGenerator
    {
        //TODO: add FillModeEnum: Constant
        public enum FillModeEnum { Nearest, Reflect };

        #region private fields
        private readonly Random[] _rands;
        //randomly shift images horizontally
        private readonly double _widthShiftRangeInPercentage;
        //randomly shift images vertically
        private readonly double _heightShiftRangeInPercentage;
        //randomly flip images
        private readonly bool _horizontalFlip;
        //randomly flip images
        private readonly bool _verticalFlip;
        //set mode for filling points outside the input boundaries
        private readonly FillModeEnum _fillMode;
        //value used for fill_mode  {get;set;} = "constant"
        private readonly double _fillModeConstantVal;
        /// <summary>
        /// % of the max(width,height) of the zero mask to apply to the input picture (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR-10 / 8/32=0.25 (= 8x8) for CIFAR-100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// less or equal to 0.0 means no cutout
        /// </summary>
        private readonly double _cutoutPatchPercentage;
        /// <summary>
        /// The alpha coefficient used for CutMix
        /// A value less or equal then 0 will disable CutMix
        /// Alpha will be used as an input of the beta law to compute lambda
        /// lambda is the % of the original to keep (1-lambda will be taken from another element and mixed with current)
        /// the % of the max(width,height) of the CutMix mask to apply to the input picture (see: https://arxiv.org/pdf/1905.04899.pdf)
        /// </summary>
        private readonly double _alphaCutMix;
        /// <summary>
        /// The alpha coefficient used for Mixup
        /// A value less or equal then 0 will disable Mixup (see: https://arxiv.org/pdf/1710.09412.pdf)
        /// </summary>
        private readonly double _alphaMixup;


        /// <summary>
        /// rotation range in degrees, in [0,180] range.
        /// The actual rotation will be a random number in [-_rotationRangeInDegrees,+_rotationRangeInDegrees]
        /// </summary>
        private readonly double _rotationRangeInDegrees;
        /// <summary>
        /// Range for random zoom. [lower, upper] = [1 - _zoomRange, 1 + _zoomRange].
        /// </summary>
        private readonly double _zoomRange;
        #endregion

        public static readonly ImageDataGenerator NoDataAugmentation = new ImageDataGenerator(0, 0, false, false, FillModeEnum.Nearest, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        public ImageDataGenerator(double widthShiftRangeInPercentage, double heightShiftRangeInPercentage,
            bool horizontalFlip, bool verticalFlip, FillModeEnum fillMode, double fillModeConstantVal,
            double cutoutPatchPercentage,
            double alphaCutMix,
            double alphaMixup,
            double rotationRangeInDegrees,
            double zoomRange)
        {
            Debug.Assert(widthShiftRangeInPercentage >= 0);
            Debug.Assert(widthShiftRangeInPercentage <= 1.0);
            Debug.Assert(heightShiftRangeInPercentage >= 0);
            Debug.Assert(heightShiftRangeInPercentage <= 1.0);
            Debug.Assert(cutoutPatchPercentage <= 1.0);
            Debug.Assert(rotationRangeInDegrees >= 0);
            Debug.Assert(rotationRangeInDegrees <= 180.0);
            Debug.Assert(zoomRange >= 0);
            Debug.Assert(zoomRange <= 1.0);
            _widthShiftRangeInPercentage = widthShiftRangeInPercentage;
            _heightShiftRangeInPercentage = heightShiftRangeInPercentage;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _fillMode = fillMode;
            _fillModeConstantVal = fillModeConstantVal;
            _cutoutPatchPercentage = cutoutPatchPercentage;
            _alphaCutMix = alphaCutMix;
            _alphaMixup = alphaMixup;
            _rotationRangeInDegrees = rotationRangeInDegrees;
            _zoomRange = zoomRange;
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }
        }


        private List<Operation> GetSubPolicy(int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch)
        {
            var rand = _rands[indexInMiniBatch % _rands.Length];
            var result = new List<Operation>();

            var miniBatchShape = xOriginalMiniBatch.Shape;
            result.Add(Operations.CutMix.ValueOf(_alphaCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Operations.Cutout.ValueOf(_cutoutPatchPercentage, rand, miniBatchShape));
            result.Add(Rotate.ValueOf(_rotationRangeInDegrees, rand, miniBatchShape));

            double widthMultiplier = 1.0;
            if (_zoomRange > 0)
            {
                //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
                var zoom = 2 * _zoomRange * rand.NextDouble() - _zoomRange;
                widthMultiplier = (1.0 + zoom);
                result.Add(new ShearX(widthMultiplier, miniBatchShape));
            }
            result.Add(TranslateX.ValueOf(_heightShiftRangeInPercentage, rand, miniBatchShape));

            if (_zoomRange > 0)
            {
                double heightMultiplier = widthMultiplier;
                result.Add(new ShearY(heightMultiplier, miniBatchShape));
            }

            var verticalFlip = _verticalFlip && rand.Next(2) == 0;
            if (verticalFlip)
            {
                result.Add(new VerticalFlip(miniBatchShape));
            }
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            if (horizontalFlip)
            {
                result.Add(new HorizontalFlip(miniBatchShape));
            }
            result.Add(Mixup.ValueOf(_alphaMixup, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.RemoveAll(x => x == null);
            return result;
        }

        public void DataAugmentationForMiniBatchV2(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch, 
            CpuTensor<float> yMiniBatch,
            Func<int, int> indexInMiniBatchToCategoryId)
        {
            var subPolicy = GetSubPolicy(indexInMiniBatch, xOriginalMiniBatch);
            SubPolicy.Apply(subPolicy, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, yMiniBatch, indexInMiniBatchToCategoryId, _fillMode);
        }

        /// <summary>
        /// takes the input picture at index 'indexInMiniBatch' in 'xOriginalMiniBatch'
        /// and stores a data augmented version of it at index 'indexInMiniBatch' of 'xTransformedMiniBatch'
        /// Field 'yMiniBatch' will only be updated for some data augmentation techniques (ex: CutMix, MixUp)
        /// </summary>
        /// <param name="indexInMiniBatch">the index of the picture to process in 'xOriginalMiniBatch' & 'xDataAugmentedMiniBatch'</param>
        /// <param name="xOriginalMiniBatch">tensor with all original pictures (not augmented) in the current mini batch
        /// The mini batch size is xOriginalMiniBatch.Shape[0] </param>
        /// <param name="xDataAugmentedMiniBatch"> tensor where the augmented pictures of the mini batch will be stored
        /// this method will only update the picture at index 'indexInMiniBatch' of 'xDataAugmentedMiniBatch'
        /// </param>
        /// <param name="yMiniBatch">expected categories of the original (not augmented) elements
        /// Some data augmentation techniques (ex: CutMix, MixUp) will require to update those expected categories
        /// </param>
        /// <param name="indexInMiniBatchToCategoryId"></param>
        public void DataAugmentationForMiniBatch(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch, 
            CpuTensor<float> yMiniBatch, 
            Func<int,int> indexInMiniBatchToCategoryId)
        {
            // NCHW tensors
            Debug.Assert(xOriginalMiniBatch.SameShape(xDataAugmentedMiniBatch));

            int miniBatchSize = xOriginalMiniBatch.Shape[0];


            var rand = _rands[indexInMiniBatch % _rands.Length];

            //random shift
            int verticalShiftRangeInPixels = GetShiftInPixel(xOriginalMiniBatch.Shape[2], _heightShiftRangeInPercentage);
            int horizontalShiftRangeInPixels = GetShiftInPixel(xOriginalMiniBatch.Shape[3], _widthShiftRangeInPercentage);
            var verticalShift = rand.Next(2 * verticalShiftRangeInPixels + 1) - verticalShiftRangeInPixels;
            var horizontalShift = rand.Next(2 * horizontalShiftRangeInPixels + 1) - horizontalShiftRangeInPixels;

            //random horizontal and vertical flip
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            var verticalFlip = _verticalFlip && rand.Next(2) == 0;

            //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
            var zoom = 2*_zoomRange * rand.NextDouble() - _zoomRange;
            var widthMultiplier = (1.0+zoom);
            var heightMultiplier = (1.0+zoom);

            var nbRows = xDataAugmentedMiniBatch.Shape[2];
            var nbCols = xDataAugmentedMiniBatch.Shape[3];
            Cutout(nbRows, nbCols, rand, out var cutoutRowStart, out var cutoutRowEnd, out var cutoutColStart, out var cutoutColEnd);
            CutMix(nbRows, nbCols, rand, out var cutMixRowStart, out var cutMixRowEnd, out var cutMixColStart, out var cutMixColEnd);
            //the index of the element in the mini batch that will be used for the CutMix with the current 'indexInMiniBatch' element
            int indexInMiniBatchForCutMix = (indexInMiniBatch+1)% miniBatchSize;
            int indexInMiniBatchForMixup = (indexInMiniBatch+2)% miniBatchSize;
            //random rotation in range [-_rotationRangeInDegrees, +_rotationRangeInDegrees]
            var rotationInDegrees = 2 * _rotationRangeInDegrees * rand.NextDouble() - _rotationRangeInDegrees;

            var mixupLambda = (_alphaMixup > 0.0) ? (float)Utils.BetaDistribution(_alphaMixup, _alphaMixup, rand) : (float?)null;

            InitializeOutputPicture(
                xOriginalMiniBatch, 
                xDataAugmentedMiniBatch, indexInMiniBatch,
                horizontalShift, verticalShift, _fillMode,
                horizontalFlip, verticalFlip,
                widthMultiplier, heightMultiplier,
                cutoutRowStart, cutoutRowEnd, cutoutColStart, cutoutColEnd,
                cutMixRowStart, cutMixRowEnd, cutMixColStart, cutMixColEnd, indexInMiniBatchForCutMix,
                mixupLambda, indexInMiniBatchForMixup,
                rotationInDegrees);

            // if CutMix has been used, wee need to update the expected output ('y' tensor)
            if (_alphaCutMix>0.0 && indexInMiniBatchToCategoryId(indexInMiniBatch) != indexInMiniBatchToCategoryId(indexInMiniBatchForCutMix))
            {
                float cutMixLambda = 1f - ((float)((cutMixRowEnd - cutMixRowStart + 1) * (cutMixColEnd - cutMixColStart + 1))) / (nbCols * nbRows);
                // We need to update the expected y using CutMix lambda
                // the associated y is:
                //        'cutMixLambda' % of the category of the element at 'indexInMiniBatch'
                //      '1-cutMixLambda' % of the category of the element at 'indexInMiniBatchForCutMix'
                yMiniBatch.Set(indexInMiniBatch, indexInMiniBatchToCategoryId(indexInMiniBatch), cutMixLambda);
                yMiniBatch.Set(indexInMiniBatch, indexInMiniBatchToCategoryId(indexInMiniBatchForCutMix), 1f - cutMixLambda);
            }

            // if MixUp has been used, wee need to update the expected output ('y' tensor)
            if (mixupLambda.HasValue && indexInMiniBatchToCategoryId(indexInMiniBatch) != indexInMiniBatchToCategoryId(indexInMiniBatchForMixup))
            {
                // We need to update the expected y using Mixup lambda
                // the associated y is:
                //        'mixupLambda' % of the category of the element at 'indexInMiniBatch'
                //      '1-mixupLambda' % of the category of the element at 'indexInMiniBatchForMixup'
                yMiniBatch.Set(indexInMiniBatch, indexInMiniBatchToCategoryId(indexInMiniBatch), mixupLambda.Value);
                yMiniBatch.Set(indexInMiniBatch, indexInMiniBatchToCategoryId(indexInMiniBatchForMixup), 1f - mixupLambda.Value);
            }

        }
        public bool Equals(ImageDataGenerator other, double epsilon, string id, ref string errors)
        {
            var equals = true;
            equals &= Utils.Equals(_widthShiftRangeInPercentage, other._widthShiftRangeInPercentage, epsilon, id + ":_widthShiftRange", ref errors);
            equals &= Utils.Equals(_heightShiftRangeInPercentage, other._heightShiftRangeInPercentage, epsilon, id + ":_heightShiftRange", ref errors);
            equals &= Utils.Equals(_horizontalFlip, other._horizontalFlip, id + ":_horizontalFlip", ref errors);
            equals &= Utils.Equals(_verticalFlip, other._verticalFlip, id + ":_verticalFlip", ref errors);
            equals &= Utils.Equals(_fillMode, other._fillMode, id + ":_fillMode", ref errors);
            equals &= Utils.Equals(_fillModeConstantVal, other._fillModeConstantVal, epsilon, id + ":_fillModeConstantVal", ref errors);
            equals &= Utils.Equals(_cutoutPatchPercentage, other._cutoutPatchPercentage, epsilon, id + ":_cutoutPatchPercentage", ref errors);
            equals &= Utils.Equals(_alphaCutMix, other._alphaCutMix, epsilon, id + ":_alphaCutMix", ref errors);
            equals &= Utils.Equals(_alphaMixup, other._alphaMixup, epsilon, id + ":_AlphaMixup", ref errors);
            equals &= Utils.Equals(_rotationRangeInDegrees, other._rotationRangeInDegrees, epsilon, id + ":_rotationRangeInDegrees", ref errors);
            equals &= Utils.Equals(_zoomRange, other._zoomRange, epsilon, id + ":_zoomRange", ref errors);
            return equals;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="xInputBufferPictures"></param>
        /// <param name="xOutputBufferPictures"></param>
        /// <param name="indexInBuffer"></param>
        /// <param name="horizontalShift"></param>
        /// <param name="verticalShift"></param>
        /// <param name="_fillMode"></param>
        /// <param name="horizontalFlip"></param>
        /// <param name="verticalFlip"></param>
        /// <param name="widthMultiplier"></param>
        /// <param name="heightMultiplier"></param>
        /// <param name="cutoutRowStart"></param>
        /// <param name="cutoutRowEnd"></param>
        /// <param name="cutoutColStart"></param>
        /// <param name="cutoutColEnd"></param>
        /// <param name="cutMixRowStart"></param>
        /// <param name="cutMixRowEnd"></param>
        /// <param name="cutMixColStart"></param>
        /// <param name="cutMixColEnd"></param>
        /// <param name="indexInMiniBatchForCutMix"></param>
        /// <param name="mixupLambda"></param>
        /// <param name="indexInMiniBatchForMixup"></param>
        /// <param name="rotationInDegrees"></param>
        public static void InitializeOutputPicture(CpuTensor<float> xInputBufferPictures,
            CpuTensor<float> xOutputBufferPictures, int indexInBuffer,
            int horizontalShift, int verticalShift, FillModeEnum _fillMode,
            bool horizontalFlip, bool verticalFlip,
            double widthMultiplier, double heightMultiplier,
            int cutoutRowStart, int cutoutRowEnd, int cutoutColStart, int cutoutColEnd,
            int cutMixRowStart, int cutMixRowEnd, int cutMixColStart, int cutMixColEnd, int indexInMiniBatchForCutMix,
            float ?mixupLambda, int indexInMiniBatchForMixup,
            double rotationInDegrees)
        {
            var nbRows = xOutputBufferPictures.Shape[2];
            var nbCols = xOutputBufferPictures.Shape[3];
            var transformer = new PointTransformer(verticalShift, horizontalShift, horizontalFlip, verticalFlip,
                widthMultiplier, heightMultiplier, rotationInDegrees, nbRows, nbCols);


            for (int channel = 0; channel < xOutputBufferPictures.Shape[1]; ++channel)
            {
                for (int rowOutput = 0; rowOutput < nbRows; ++rowOutput)
                {
                    var outputPictureIdx = xOutputBufferPictures.Idx(indexInBuffer, channel, rowOutput, 0);
                    for (int colOutput = 0; colOutput < nbCols; ++colOutput)
                    {
                        //we check if we should apply cutMix to the pixel
                        //this CutMix check must be performed *before* the Cutout check (below)
                        if (rowOutput >= cutMixRowStart && rowOutput <= cutMixRowEnd && colOutput >= cutMixColStart && colOutput <= cutMixColEnd)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = xInputBufferPictures.Get(indexInMiniBatchForCutMix, channel, rowOutput, colOutput);
                            continue;
                        }

                        //we check if we should apply Cutout to the pixel
                        //this Cutout check must be performed *after* the CutMix check (above)
                        if (rowOutput >= cutoutRowStart && rowOutput <= cutoutRowEnd && colOutput >= cutoutColStart && colOutput <= cutoutColEnd)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = 0;
                            continue;
                        }

                        var rowInput = transformer.UnconvertRow(rowOutput, colOutput);
                        if (rowInput < 0)
                        {
                            rowInput = _fillMode == FillModeEnum.Reflect ? Math.Abs(rowInput + 1) : 0;
                        }
                        if (rowInput >= nbRows)
                        {
                            rowInput = _fillMode == FillModeEnum.Reflect ? (nbRows - 1 - (rowInput - nbRows)) : (nbRows - 1);
                        }
                        rowInput = Math.Min(Math.Max(0, rowInput), nbRows - 1);
                        Debug.Assert(rowInput >= 0 && rowInput < nbRows);

                        var colInput = transformer.UnconvertCol(rowOutput, colOutput);
                        if (colInput < 0)
                        {
                            colInput = _fillMode == FillModeEnum.Reflect ? Math.Abs(colInput + 1) : 0;
                        }
                        if (colInput >= nbCols)
                        {
                            colInput = _fillMode == FillModeEnum.Reflect ? (nbCols - 1 - (colInput - nbCols)) : (nbCols - 1);
                        }
                        colInput = Math.Min(Math.Max(0, colInput), nbCols - 1);
                        Debug.Assert(colInput >= 0 && colInput < nbCols);

                        var valueInOriginalPicture = xInputBufferPictures.Get(indexInBuffer, channel, rowInput, colInput);
                        xOutputBufferPictures[outputPictureIdx + colOutput] = valueInOriginalPicture;
                        if (mixupLambda.HasValue)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = 
                                        mixupLambda.Value * valueInOriginalPicture
                                + (1 - mixupLambda.Value) * xInputBufferPictures.Get(indexInMiniBatchForMixup, channel,rowOutput, colOutput);
                        }
                    }
                }
            }
        }


        #region serialization
        public string Serialize()
        {
            if (!UseDataAugmentation)
            {
                return "";
            }
            return new Serializer()
                .Add(nameof(_widthShiftRangeInPercentage), _widthShiftRangeInPercentage)
                .Add(nameof(_heightShiftRangeInPercentage), _heightShiftRangeInPercentage)
                .Add(nameof(_horizontalFlip), _horizontalFlip)
                .Add(nameof(_verticalFlip), _verticalFlip)
                .Add(nameof(_fillMode), (int)_fillMode)
                .Add(nameof(_fillModeConstantVal), _fillModeConstantVal)
                .Add(nameof(_cutoutPatchPercentage), _cutoutPatchPercentage)
                .Add(nameof(_alphaCutMix), _alphaCutMix)
                .Add(nameof(_alphaMixup), _alphaMixup)
                .Add(nameof(_rotationRangeInDegrees), _rotationRangeInDegrees)
                .Add(nameof(_zoomRange), _zoomRange)
                .ToString();
        }
        public static ImageDataGenerator ValueOf(IDictionary<string, object> serialized)
        {
            if (!serialized.ContainsKey(nameof(_widthShiftRangeInPercentage)))
            {
                return NoDataAugmentation;
            }
            return new ImageDataGenerator(
                (double)serialized[nameof(_widthShiftRangeInPercentage)],
                (double)serialized[nameof(_heightShiftRangeInPercentage)],
                (bool)serialized[nameof(_horizontalFlip)],
                (bool)serialized[nameof(_verticalFlip)],
                (FillModeEnum)serialized[nameof(_fillMode)],
                (double)serialized[nameof(_fillModeConstantVal)],
                (double)serialized[nameof(_cutoutPatchPercentage)],
                (double)serialized[nameof(_alphaCutMix)],
                (double)serialized[nameof(_alphaMixup)], 
                (double)serialized[nameof(_rotationRangeInDegrees)], 
                (double)serialized[nameof(_zoomRange)]);
        }
        #endregion
 
        private void Cutout(int nbRows, int nbCols, Random rand, out int rowStart, out int rowEnd, out int colStart, out int colEnd)
        {
            if (_cutoutPatchPercentage <= 0)
            {
                rowStart = rowEnd = colStart = colEnd = -1;
                return;
            }
            if (_cutoutPatchPercentage > 1.0)
            {
                throw new ArgumentException("invalid _cutoutPatchPercentage:" + _cutoutPatchPercentage);
            }

            int cutoutPatchLength = (int)Math.Round(_cutoutPatchPercentage * Math.Max(nbRows, nbCols), 0.0);

            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);

            //test on 12-aug-2019 : -46bps
            //Cutout of always the max dimension
            //rowMiddle = (cutoutPatchLength/2)+rand.Next(nbRows- cutoutPatchLength);
            //colMiddle = (cutoutPatchLength/2) + rand.Next(nbCols - cutoutPatchLength);

            //Tested on 10-aug-2019: -10bps
            //rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            //rowEnd = Math.Min(nbRows - 1, rowMiddle + cutoutPatchLength/2 - 1);
            //colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            //colEnd = Math.Min(nbCols - 1, colMiddle + cutoutPatchLength/2 - 1);

            rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            rowEnd = Math.Min(nbRows - 1, rowStart + cutoutPatchLength - 1);
            colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            colEnd = Math.Min(nbCols - 1, colStart + cutoutPatchLength - 1);
        }

        /// <summary>
        /// retrieve the area where we should put a part of another (random) element to mix it up with the current one
        /// </summary>
        /// <param name="nbRows"></param>
        /// <param name="nbCols"></param>
        /// <param name="rand"></param>
        /// <param name="rowStart"></param>
        /// <param name="rowEnd"></param>
        /// <param name="colStart"></param>
        /// <param name="colEnd"></param>
        private void CutMix(int nbRows, int nbCols, Random rand, out int rowStart, out int rowEnd, out int colStart, out int colEnd)
        {
            if (_alphaCutMix<=0.0)
            {
                rowStart = rowEnd = colStart = colEnd = -1;
                return;
            }

            //CutMix V2 : we ensure that we keep at least 50% of the original image when mixing with another one
            //validated on 18-aug-2019
            var lambda = 0.5 + 0.5 * (float)Utils.BetaDistribution(_alphaCutMix, _alphaCutMix, rand);
            //var lambda = (float)Utils.BetaDistribution(_alphaCutMix, _alphaCutMix, rand);

            var cutMixHeight = (int)(nbRows * Math.Sqrt(1.0 - lambda));
            var cutMixWidth = (int)(nbCols * Math.Sqrt(1.0 - lambda));

            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);
            rowStart = Math.Max(0, rowMiddle - cutMixHeight / 2);
            rowEnd = Math.Min(nbRows - 1, rowMiddle + cutMixHeight/2 - 1);
            colStart = Math.Max(0, colMiddle - cutMixWidth/2 / 2);
            colEnd = Math.Min(nbCols - 1, colMiddle + cutMixWidth/2 - 1);
        }
        public bool UseDataAugmentation => !ReferenceEquals(this, NoDataAugmentation);
        private static int GetShiftInPixel(int pictureWidth, double widthShiftRange)
        {
            if (widthShiftRange <= 0)
            {
                return 0;
            }
            return (int)Math.Ceiling(pictureWidth * widthShiftRange);
        }
    }
}
