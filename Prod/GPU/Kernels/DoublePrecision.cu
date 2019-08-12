extern "C" {

    __global__ void Sum(int N, const double* __restrict left, const double* __restrict right, double* __restrict output) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
			output[i] = left[i] + right[i];
	}

    __global__ void UpdateAdamOptimizer(int N, double beta1, double beta2, double epsilon, double multiplicative_factor,
				const double* __restrict dW, double* __restrict W,
				double* __restrict adam_vW, double* __restrict adam_sW) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			double dw = dW[i];
			adam_vW[i] = beta1*adam_vW[i]+(1-beta1)*dw;
            adam_sW[i] = beta2*adam_sW[i]+(1-beta2)*dw*dw;
			W[i] -= (multiplicative_factor * adam_vW[i]) / (sqrt(adam_sW[i]) + epsilon);
		}
	}

	__global__ void ComputeAccuracy(int N, int categoryCount, double *countOk, const double* __restrict yExpectedOneHot, const double* __restrict yPredicted) 	{
		int i = blockIdx.x * blockDim.x + threadIdx.x; 
		if (i < N) {
			if (categoryCount == 1)
			{
				float error = fabsf(yExpectedOneHot[i] - yPredicted[i]);
				countOk[i] = (error < 0.5) ? 1.0 : 0.0;
				return;
			}

			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndexPredicted = startIndex;
			int maxIndexExpected = startIndex;
			for (int j = startIndex+1; j < endIndexExcluded; ++j)
			{
				if (yPredicted[j] > yPredicted[maxIndexPredicted])
					maxIndexPredicted = j;
				if (yExpectedOneHot[j] > yExpectedOneHot[maxIndexExpected])
					maxIndexExpected = j;
			}
			countOk[i] = (maxIndexPredicted == maxIndexExpected) ? 1.0f : 0.0f;
		}
	}

	__global__ void ComputeCategoricalCrossentropyLoss(int N, int categoryCount, double *losses, const double* __restrict yExpectedOneHot, const double* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			double loss = 0.0;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndex = startIndex;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				double predicted = yPredicted[j];
				double expected = yExpectedOneHot[j];
				if (predicted > 0)
					loss -= expected * logf(predicted);
			}
			losses[i] = loss;
		}
	}

	__global__ void ComputeBinaryCrossentropyLoss(int N, int categoryCount, double *losses, const double* __restrict yExpectedOneHot, const double* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			double loss = 0.0;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndex = startIndex;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				double predicted = yPredicted[j];
				double expected = yExpectedOneHot[j];
				if ((predicted > 0.0)&&(predicted<1.0))
					loss -= (expected*logf(predicted) + (1.0-expected)*logf(1.0-predicted))/ categoryCount;
			}
			losses[i] = loss;
		}
	}

	__global__ void Concatenate(int N, int m, double* __restrict concat, int concatMultDim0, const double* __restrict a, int aMultDim0, const double* __restrict b, int bMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		concat[i] = (colInConcat<aMultDim0)?a[row*aMultDim0+colInConcat]:b[row*bMultDim0+colInConcat-aMultDim0];
	}

	__global__ void Split(int N, int m, const double* __restrict concat, int concatMultDim0, double* __restrict a, int aMultDim0, double* __restrict b, int bMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		if (colInConcat<aMultDim0)
			a[row*aMultDim0+colInConcat] = concat[i];
		else
			b[row*bMultDim0+colInConcat-aMultDim0] = concat[i];
	}
}
