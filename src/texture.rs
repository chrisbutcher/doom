use image::GenericImageView;
use std::path::Path;

pub struct Texture {
  pub texture: wgpu::Texture,
  pub view: wgpu::TextureView,
  pub sampler: wgpu::Sampler,
}

impl Texture {
  pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

  pub fn load<P: AsRef<Path>>(device: &wgpu::Device, path: P) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
    // Needed to appease the borrow checker
    let path_copy = path.as_ref().to_path_buf();
    let label = path_copy.to_str();
    let img = image::open(path)?;
    Self::from_image(device, &img, label)
  }

  pub fn create_depth_texture(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, label: &str) -> Self {
    let size = wgpu::Extent3d {
      width: sc_desc.width,
      height: sc_desc.height,
      depth: 1,
    };
    let desc = wgpu::TextureDescriptor {
      label: Some(label),
      size,
      array_layer_count: 1,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: Self::DEPTH_FORMAT,
      usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_SRC,
    };
    let texture = device.create_texture(&desc);

    let view = texture.create_default_view();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Nearest,
      min_filter: wgpu::FilterMode::Nearest,
      mipmap_filter: wgpu::FilterMode::Nearest,
      lod_min_clamp: -100.0,
      lod_max_clamp: 100.0,
      compare: wgpu::CompareFunction::LessEqual,
    });

    Self { texture, view, sampler }
  }

  #[allow(dead_code)]
  pub fn from_bytes(
    device: &wgpu::Device,
    bytes: &[u8],
    label: &str,
  ) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
    let img = image::load_from_memory(bytes)?;
    Self::from_image(device, &img, Some(label))
  }

  pub fn from_image(
    device: &wgpu::Device,
    img: &image::DynamicImage,
    label: Option<&str>,
  ) -> Result<(Self, wgpu::CommandBuffer), failure::Error> {
    let rgba = img.to_rgba();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
      width: dimensions.0,
      height: dimensions.1,
      depth: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label,
      size,
      array_layer_count: 1,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba8UnormSrgb,
      usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });

    let buffer = device.create_buffer_with_data(&rgba, wgpu::BufferUsage::COPY_SRC);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("texture_buffer_copy_encoder"),
    });

    encoder.copy_buffer_to_texture(
      wgpu::BufferCopyView {
        buffer: &buffer,
        offset: 0,
        bytes_per_row: 4 * dimensions.0,
        rows_per_image: dimensions.1,
      },
      wgpu::TextureCopyView {
        texture: &texture,
        mip_level: 0,
        array_layer: 0,
        origin: wgpu::Origin3d::ZERO,
      },
      size,
    );

    let cmd_buffer = encoder.finish();

    let view = texture.create_default_view();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Nearest,
      min_filter: wgpu::FilterMode::Nearest,
      mipmap_filter: wgpu::FilterMode::Nearest,
      lod_min_clamp: -100.0,
      lod_max_clamp: 100.0,
      compare: wgpu::CompareFunction::Always,
    });

    Ok((Self { texture, view, sampler }, cmd_buffer))
  }
}
