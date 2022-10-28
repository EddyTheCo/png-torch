#include"utils/png-torch.hpp"
#include <png.h>

namespace custom_models{
namespace png_interface{

	void write_png_file(std::string filename, const torch::Tensor &image_ /*{batch,width,height} batch(x*x)*/) {


	torch::Tensor image=image_.mul(256).to(torch::kUInt8);


	FILE *fp = fopen(filename.c_str(), "wb");



	std::vector<unsigned char*> row_pointers;
	std::vector<torch::Tensor> container;
	for(auto i=0;i<std::sqrt(image.size(0))*image.size(1);i++)
	{

		auto T=image.index(
				{torch::indexing::Slice({(i/image.size(1))*std::sqrt(image.size(0)), (i/image.size(1)+1)*std::sqrt(image.size(0))}),
				torch::indexing::Slice({i%image.size(1),i%image.size(1)+1}),
				torch::indexing::Slice()}
				).contiguous().to(torch::kCPU);

		container.push_back(T);
		row_pointers.push_back(container.back().data_ptr<unsigned char>());
	}

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);


	png_infop info = png_create_info_struct(png);


	png_init_io(png, fp);

	// Output is 8bit depth, RGBA format.
	png_set_IHDR(
			png,
			info,
			sqrt(image.size(0))*image.size(1), sqrt(image.size(0))*image.size(2),
			8,
			PNG_COLOR_TYPE_GRAY,
			PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_DEFAULT,
			PNG_FILTER_TYPE_DEFAULT
		    );
	png_write_info(png, info);

	png_write_image(png, row_pointers.data());

	png_write_end(png, NULL);

	fclose(fp);

	png_destroy_write_struct(&png, &info);
}
}
}
