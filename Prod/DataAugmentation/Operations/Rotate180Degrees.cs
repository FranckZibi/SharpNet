namespace SharpNet.DataAugmentation.Operations
{
    public class Rotate180Degrees : Operation
    {
        private readonly int _nbRows;
        private readonly int _nbCols;

        public Rotate180Degrees(int nbRows, int nbCols)
        {
            _nbRows = nbRows;
            _nbCols = nbCols;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (_nbRows - row - 1, _nbCols - col-1);
        }
    }
}
