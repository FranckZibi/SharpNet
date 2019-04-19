extern "C" {

    __global__ void Sum(int N, const float* __restrict left, const float* __restrict right, float* __restrict output) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
			output[i] = left[i] + right[i];
	}

    __global__ void UpdateAdamOptimizer(int N, float beta1, float beta2, float epsilon, float multiplicative_factor,
				const float* __restrict dW, float* __restrict W,
				float* __restrict adam_vW, float* __restrict adam_sW) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			float dw = dW[i];
			adam_vW[i] = beta1*adam_vW[i]+(1-beta1)*dw;
            adam_sW[i] = beta2*adam_sW[i]+(1-beta2)*dw*dw;
			W[i] -= (multiplicative_factor * adam_vW[i]) / (sqrtf(adam_sW[i]) + epsilon);
		}
	}

	//TODO remove this function
	__global__ void UpdateSGDOptimizer(int N, float learningRate, float momentum, float decay, bool usenesterov,
		const float* __restrict dW, float* __restrict W, float* __restrict velocity) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			float dw = dW[i];
			velocity[i] = (momentum * velocity[i]) - (dw * learningRate);
			if (usenesterov)
			{
				W[i] += momentum*velocity[i] - (dw * learningRate);
			}
			else
			{
				W[i] += velocity[i];
			}

		}
	}

	__global__ void ComputeAccuracy(int N, int categoryCount, float *countOk, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted) 
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			if (categoryCount == 1)
			{
				float error = fabsf(yExpectedOneHot[i] - yPredicted[i]);
				countOk[i] = (error < 0.5f) ? 1.0f : 0.0f;
				return;
			}

			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndex = startIndex;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				if (yPredicted[j] > yPredicted[maxIndex])
					maxIndex = j;
			}
			countOk[i] = (yExpectedOneHot[maxIndex] > 0.9f) ? 1.0f : 0.0f;
		}
	}

	__global__ void ComputeCategoricalCrossentropyLoss(int N, int categoryCount, float *losses, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndex = startIndex;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				if (predicted > 0)
					loss -= expected * logf(predicted);
			}
			losses[i] = loss;
		}
	}

	__global__ void ComputeBinaryCrossentropyLoss(int N, int categoryCount, float *losses, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndex = startIndex;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				//if ((predicted>0.01)&&(predicted<0.99f))
				if ((predicted>0.0f)&&(predicted<1.0f))
					loss -= (expected*logf(predicted) + (1.0f-expected)*logf(1.0f-predicted))/ categoryCount;
			}
			losses[i] = loss;
		}
	}


}