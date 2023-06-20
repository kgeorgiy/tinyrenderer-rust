use crate::egl::{Color, scale_color, Shader, transform_vector};
use crate::linear::{Matr, Vect};
use crate::texture::Texture;

pub struct ShaderSettings<'t> {
    pub texture: Texture<Color>,
    pub normals: Normals,
    pub solid_color: Option<Color>,
    pub cell_shading: bool,
    pub light: Light<'t>,
}

pub struct UniversalShader<'t> {
    settings: &'t ShaderSettings<'t>,
    light: Light<'t>,
    uniform_transform: Matr<4, 4>,
    uniform_transform_it: Matr<4, 4>,
    varying_uv: Matr<2, 3>,
    varying_normal: Matr<3, 3>,
    device: [Vect<3>; 3],
}

impl<'t> UniversalShader<'t> {
    pub fn new(settings: &'t ShaderSettings) -> Self {
        UniversalShader {
            settings,
            light: Light::Ambient,
            uniform_transform: Matr::identity(),
            uniform_transform_it: Matr::identity(),
            varying_uv: Matr::zero(),
            device: [Vect::ZERO; 3],
            varying_normal: Matr::zero(),
        }
    }

    fn darboux(&self, bc: &Vect<3>) -> Matr<3, 3> {
        let n = &self.varying_normal * bc;
        let dev = &self.device;
        let from_darboux = &Matr::from([n, dev[1] - dev[0], dev[2] - dev[0]]);
        let to_darboux = &from_darboux.inverse();
        Matr::from_columns([self.darb(to_darboux, 0), self.darb(to_darboux, 1), n])
    }

    fn darb(&self, to_darboux: &Matr<3, 3>, index: usize) -> Vect<3> {
        let uv = self.varying_uv.get_row(index);
        (to_darboux * &(uv - Vect::ONES * uv.x())).normalize()
    }

    pub fn transform_normal(&self, normal: &Vect<3>) -> Vect<3> {
        transform_vector(&self.uniform_transform_it, normal)
    }
}

impl<'t> Shader<Color> for UniversalShader<'t> {
    fn set_transforms(&mut self, transform: &Matr<4, 4>, light_transform: &Matr<3, 3>) {
        self.uniform_transform = transform.clone();
        self.uniform_transform_it = transform.inverse_transpose();
        self.light = self.settings.light.transform(&(&transform.proj::<3, 3>() * light_transform));
    }

    fn vertex(&mut self, index: usize, vertex: &Vect<3>, normal: &Vect<3>, texture: &Vect<3>, _screen: &Vect<4>) {
        self.varying_uv.set_col(index, &texture.proj());
        self.device[index] = Vect::point(&(&self.uniform_transform * &vertex.to_point()));
        self.varying_normal.set_col(index, &self.transform_normal(normal));
    }

    fn fragment(&mut self, bc: &Vect<3>) -> Color {
        let uv = &(&self.varying_uv * bc);
        let settings = &self.settings;

        let normal: &Vect<3> = &match &settings.normals {
            Normals::Flat => self.varying_normal * Vect::ONES / 3.0,
            Normals::Phong => &self.varying_normal * bc,
            Normals::Global(normals) => self.transform_normal(normals.get(uv)),
            Normals::Tangent(normals) => &self.darboux(bc) * normals.get(uv),
        }.normalize();

        let intensity = self.light.intensity(uv, normal);
        let pixel = settings.solid_color.unwrap_or(*settings.texture.get(uv));
        scale_color(pixel, if settings.cell_shading { (intensity * 6.0).round() / 6.0 } else { intensity })
    }
}

#[allow(dead_code)]
pub enum Normals {
    Flat,
    Phong,
    Global(Texture<Vect<3>>),
    Tangent(Texture<Vect<3>>),
}

#[allow(dead_code)]
pub enum Light<'t> {
    Ambient,
    Diffuse(Vect<3>),
    Specular(Vect<3>, &'t Texture<f32>),
    Composite(f32, Vec<(Light<'t>, f32)>),
}

impl<'t> Light<'t> {
    pub fn intensity(&self, uv: &Vect<2>, normal: &Vect<3>) -> f32 {
        match self {
            Light::Ambient => 1.0,
            Light::Diffuse(light) => normal * light,
            Light::Specular(light, specular) => {
                let reflected = (normal * (normal * light * 2.0) - *light).normalize();
                reflected.z().max(0.0).powf(5.0 + specular.get(uv))
            }
            Light::Composite(div, parts) => {
                let mut result = 0.0;
                for (item, intensity) in parts {
                    result += item.intensity(uv, normal) * intensity;
                }
                result / div
            }
        }.max(0.0)
    }

    pub fn transform(&self, transform: &Matr<3, 3>) -> Self {
        match self {
            Light::Ambient => Light::Ambient,
            Light::Diffuse(light) => Light::Diffuse(transform * light),
            Light::Specular(light, texture) => Light::Specular(
        transform * light,
                texture
            ),
            Light::Composite(div, parts) => Light::Composite(
                *div,
                parts.iter()
                    .map(|(part, intensity)| (part.transform(transform), *intensity))
                    .collect()
            ),
        }
    }
}
