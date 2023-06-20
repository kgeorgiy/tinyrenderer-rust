use crate::egl::Shader;
use crate::linear::{Matr, Vect};

pub struct DepthShader {
    screen_varying: Matr<3, 3>,
}

impl DepthShader {
    pub(crate) fn new() -> Self {
        DepthShader { screen_varying: Matr::zero() }
    }
}

impl Shader<f32> for DepthShader {
    fn set_transforms(&mut self, _transform: &Matr<4, 4>, _light_transform: &Matr<3, 3>) {
    }

    fn vertex(&mut self, index: usize, _vertex: &Vect<3>, _normal: &Vect<3>, _texture: &Vect<3>, screen: &Vect<4>) {
        self.screen_varying.set_col(index, &Vect::point(screen));
    }

    fn fragment(&mut self, bc: &Vect<3>) -> f32 {
        let depth = (&self.screen_varying * bc).z();
        println!("{:.3}", depth);
        depth / 100.0
    }
}
