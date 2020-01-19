
namespace SharpNet.DataAugmentation.Operations
{
    public class VerticalFlip : Operation
    {
        private readonly int _nbRows;

        public VerticalFlip(int nbRows)
        {
            _nbRows = nbRows;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (_nbRows - row - 1, col);
        }
        public override bool ChangeCoordinates()
        {
            return true;
        }

    }
}