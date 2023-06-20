use std::error::Error;
use std::fmt::{Display, Formatter};
use std::result::Result;

use image::{Pixel, Rgb};

use crate::linear::{Matr, Vect};

pub type Color = Rgb<f32>;

pub const fn rgb_color(r: f32, g: f32, b: f32) -> Color {
    Rgb([r, g, b])
}

pub fn scale_color(pixel: Color, intensity: f32) -> Color {
    pixel.map(|c| c * intensity)
}


pub fn look_at(eye: &Vect<3>, center: &Vect<3>, up: &Vect<3>) -> Matr<4, 4> {
    let z = &(center - eye).normalize();
    let x = &(up ^ z).normalize();
    let y = &(x ^ z).normalize();
    let rotate = Matr::from([
        [x.x(), x.y(), x.z(), 0.0],
        [y.x(), y.y(), y.z(), 0.0],
        [z.x(), z.y(), z.z(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let mut translate = Matr::identity();
    translate.set_col(3, &eye.extend(1.0));

    rotate * translate
}

pub fn viewport(x: f32, y: f32, width: f32, height: f32, scale: f32) -> Matr<4, 4> {
    let depth = 255.0;

    let mut viewport = Matr::identity();
    viewport.set(0, 0, scale / 2.0);
    viewport.set(1, 1, scale / 2.0);
    viewport.set(2, 2, depth / 2.0);

    viewport.set(0, 3, x + width / 2.0);
    viewport.set(1, 3, y + height / 2.0);
    viewport.set(2, 3, depth / 2.0);

    viewport
}

pub fn projection(camera_z: f32) -> Matr<4, 4> {
    let mut projection = Matr::identity();
    projection.set(3, 2, 1.0 / camera_z);
    projection
}


pub fn transform_vector(transform: &Matr<4, 4>, vect: &Vect<3>) -> Vect<3> {
    (transform * &vect.to_vector()).proj::<3>().normalize()
}


pub type EGLResult<T> = Result<T, EGLError>;

#[derive(Debug)]
pub struct EGLError {
    pub message: String,
}

impl Display for EGLError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "EGLError: {}", self.message)
    }
}

pub fn to_egle<'t, E: Error>(context: &'t str) -> Box<dyn Fn(E) -> EGLError + 't> {
    Box::new(move |error: E| EGLError { message: format!("{}: {}", context, error) })
}

impl Error for EGLError {}


pub trait Shader<T> {
    fn set_transforms(&mut self, transform: &Matr<4, 4>, light_transform: &Matr<3, 3>);
    fn vertex(&mut self, index: usize, vertex: &Vect<3>, normal: &Vect<3>, texture: &Vect<3>, screen: &Vect<4>);
    fn fragment(&mut self, bc: &Vect<3>) -> T;
}

