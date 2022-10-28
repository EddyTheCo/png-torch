#include <ATen/ATen.h>
#include <torch/torch.h>

namespace custom_models{
namespace png_interface{
	/**
	 * Creates a image from a tensor image(batch,with,height);
	 * The resulting image will be a square with size sqrt(batch);
	 *@param filename the resulting file name;
	 *@param image tensor representing a batched image
	 *
	 */
	void write_png_file(std::string filename, const torch::Tensor &image_ /*{batch,width,height} batch(x*x)*/); 
}

}
