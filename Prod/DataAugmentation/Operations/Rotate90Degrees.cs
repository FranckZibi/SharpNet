using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class Rotate90Degrees : Operation
    {
        private readonly int _nbRows;
        private readonly int _nbCols;

        public Rotate90Degrees(int nbRows, int nbCols)
        {
            if (nbRows != nbCols)
            {
                throw new ArgumentException($"{nbRows} != {nbCols}");
            }
            _nbRows = nbRows;
            _nbCols = nbCols;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (col, _nbCols - row - 1);
        }
    }
}
