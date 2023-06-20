use image::{DynamicImage, io::Reader as ImageReader, Rgb32FImage, RgbImage};

use crate::egl::{Color, EGLResult, to_egle};
use crate::linear::Vect;

pub struct Texture<T> {
    width: f32,
    height: f32,
    data: Vec<Vec<T>>,
}

impl<T: Copy> Texture<T> {
    pub fn new(width: usize, height: usize, value: T) -> Self {
        Texture {
            width: width as f32,
            height: height as f32,
            data: (0..width).map(move |_| vec![value; height]).collect()
        }
    }

    fn load(path: String, f: fn(&Color) -> T) -> EGLResult<Self> {
        let image = ImageReader::open(path.as_str())
            .map_err(to_egle(format!("Open image '{}'", path).as_str()))?
            .decode().map_err(to_egle(format!("Decode image '{}'", path).as_str()))?
            .to_rgb32f();
        Ok(Texture {
            width: image.width() as f32,
            height: image.height() as f32,
            data: image.pixels().map(f).collect::<Vec<T>>()
                .chunks(image.width() as usize)
                .rev()
                .map(|row| row.to_vec())
                .collect()
        })
    }

    pub fn to_rgb8_helper(&self, f: fn(&T) -> [f32; 3]) -> RgbImage {
        DynamicImage::ImageRgb32F(Rgb32FImage::from_vec(
            self.width() as u32,
            self.height() as u32,
            self.data.iter().flatten().map(f).flatten().collect()
        ).unwrap()).into_rgb8()
    }


    #[inline(always)]
    pub fn get(&self, uv: &Vect<2>) -> &T {
        assert!(0.0 <= uv.x() && uv.x() < 1.0 && 0.0 <= uv.y() && uv.y() < 1.0);
        &self.data[(self.height * uv.y()) as usize][(self.width * uv.x()) as usize]
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn fill(&mut self, color: T) {
        for i in 0..self.data.len() {
            self.data[i].fill(color);
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.data[y][x] = value;
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width as usize, self.height as usize)
    }
}

impl Texture<Color> {
    pub fn load_texture(path: String) -> EGLResult<Self> {
        Self::load(path, |c| *c)
    }

    pub fn to_rgb8(&self) -> RgbImage {
        self.to_rgb8_helper(|c| c.0)
    }
}

impl Texture<Vect<3>> {
    pub fn load_normals(path: String) -> EGLResult<Self> {
        Self::load(path, |c| Vect::from(c.0))
    }
}

impl Texture<f32> {
    pub fn load_specular(path: String) -> EGLResult<Self> {
        Self::load(path, |c| c.0[0])
    }

    pub fn to_rgb8(&self) -> RgbImage {
        self.to_rgb8_helper(|c| [*c, *c, *c])
    }
}
