using System;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class PointTransformer
    {
        #region Private fields

        private readonly int _verticalShift;
        private readonly int _horizontalShift;
        private readonly bool _horizontalFlip;
        private readonly bool _verticalFlip;
        private readonly double _widthMultiplier;
        private readonly double _heightMultiplier;
        private readonly double _rotationInDegrees;
        private readonly int _nbRowsBefore;
        private readonly int _nbColsBefore;
        private readonly double _unconvertAX;
        private readonly double _unconvertBX;
        private readonly double _unconvertCX;
        private readonly double _unconvertAY;
        private readonly double _unconvertBY;
        private readonly double _unconvertCY;

        #endregion

        public PointTransformer(
            int verticalShift, int horizontalShift,
            bool horizontalFlip, bool verticalFlip,
            double widthMultiplier, double heightMultiplier,
            double rotationInDegrees,
            int nbRowsBefore, int nbColsBefore)
        {
            _verticalShift = verticalShift;
            _horizontalShift = horizontalShift;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _widthMultiplier = widthMultiplier;
            _heightMultiplier = heightMultiplier;
            _rotationInDegrees = rotationInDegrees;
            _nbRowsBefore = nbRowsBefore;
            _nbColsBefore = nbColsBefore;

            _unconvertCX = Unconvert_Slow(0, 0).row;
            _unconvertAX = (Unconvert_Slow(1, 0).row - _unconvertCX);
            _unconvertBX = (Unconvert_Slow(0, 1).row - _unconvertCX);
            _unconvertCX += 1e-8;

            _unconvertCY = Unconvert_Slow(0, 0).col;
            _unconvertAY = (Unconvert_Slow(1, 0).col - _unconvertCY);
            _unconvertBY = (Unconvert_Slow(0, 1).col - _unconvertCY);
            _unconvertCY += 1e-8;
        }

        public int UnconvertRow(int row, int col)
        {
            return CoordinateToColumnIndex(_unconvertAX * row + _unconvertBX * col + _unconvertCX);
        }

        public int UnconvertCol(int row, int col)
        {
            return CoordinateToColumnIndex(_unconvertAY * row + _unconvertBY * col + _unconvertCY);
        }

        private (double row, double col) Unconvert_Slow(int row, int col)
        {
            if (_horizontalFlip)
            {
                col = _nbColsBefore - col - 1;
            }

            if (_verticalFlip)
            {
                row = _nbRowsBefore - row - 1;
            }

            double rowBefore = (row - _verticalShift) / _heightMultiplier;
            double colBefore = (col - _horizontalShift) / _widthMultiplier;
            if (_rotationInDegrees != 0)
            {
                (rowBefore, colBefore) = XYCoordinateRotator.RotateInTopLeftReferential(rowBefore, colBefore,
                    _nbRowsBefore, _nbColsBefore, _nbRowsBefore, _nbColsBefore, -_rotationInDegrees);
            }

            return (rowBefore, colBefore);
        }

        private static int CoordinateToColumnIndex(double coordinate)
        {
            return (int) Math.Floor(coordinate);
        }
    }
}