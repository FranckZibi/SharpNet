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

	__global__ void UpdateSGDOptimizer(int N, double learningRate, double momentum, double decay, bool usenesterov,
		const double* __restrict dW, double* __restrict W, double* __restrict velocity) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			double dw = dW[i];
			velocity[i] = (momentum * velocity[i]) - (dw * learningRate);
			if (usenesterov)
			{
				W[i] += momentum * velocity[i] - (dw * learningRate);
			}
			else
			{
				W[i] += velocity[i];
			}

		}
	}
	
}