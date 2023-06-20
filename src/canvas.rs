use image::RgbImage;
use crate::egl::{Color, EGLResult, Shader, to_egle, viewport};
use crate::linear::{Matr, Vect};
use crate::texture::Texture;

pub struct CanvasSettings<T> {
    pub axes: Option<(T, T, T)>,
    pub wireframe: Option<T>,
    pub normals: Option<T>,
    pub tangent_normals: Option<(T, T, T)>,
}

impl<T> CanvasSettings<T> {
    pub fn new() -> Self {
        CanvasSettings {
            axes: None,
            wireframe: None,
            normals: None,
            tangent_normals: None,
        }
    }
}

pub struct Canvas<T> {
    texture: Texture<T>,
    z_buffer: Vec<f32>,
    fill: T,
    pub settings: CanvasSettings<T>,
}

impl<T: Copy> Canvas<T> {
    pub fn new(width: usize, height: usize, fill: T, settings: CanvasSettings<T>) -> Self {
        Canvas {
            texture: Texture::new(width, height, fill),
            z_buffer: vec![-f32::INFINITY; width * height],
            settings,
            fill,
        }
    }

    pub fn width(&self) -> f32 {
        self.texture.width()
    }

    pub fn height(&self) -> f32 {
        self.texture.height()
    }

    pub fn viewport(&self, scale: f32) -> Matr<4, 4> {
        let height = self.height();
        viewport(0.0, 0.0, self.width(), height, height * scale)
    }

    pub fn clear(&mut self) {
        self.texture.fill(self.fill);
        self.z_buffer.fill(-1e10f32);
    }

    pub fn draw_line_transformed(&mut self, color: T, transform: &Matr<4, 4>, p1: &Vect<3>, p2: &Vect<3>) {
        self.draw_line(color, &Vect::point(&(transform * &p1.to_point())), &Vect::point(&(transform * &p2.to_point())));
    }

    pub fn draw_line(&mut self, color: T, p1: &Vect<3>, p2: &Vect<3>) {
        self.draw_line_i(color, p1.x() as i32, p1.y() as i32, p2.x() as i32, p2.y() as i32)
    }

    fn draw_line_i(&mut self, color: T, x1: i32, y1: i32, x2: i32, y2: i32) {
        let (width, height) = self.texture.dimensions();
        if (x2 - x1).abs() >= (y2 - y1).abs() {
            let mut set = |u: usize, v: usize| { self.texture.set(u, v, color.clone()); };
            draw_line_helper(x1, y1, x2, y2, width, height, &mut set);
        } else {
            let mut set = |u: usize, v: usize| { self.texture.set(v, u, color.clone()); };
            draw_line_helper(y1, x1, y2, x2, height, width, &mut set);
        }
    }

    pub fn triangle(&mut self, screen: [Vect<4>; 3], shader: &mut dyn Shader<T>) {
        let settings = &self.settings;

        let vs = screen.map(|p| Vect::point(&p));

        if let Some(color) = settings.wireframe {
            for i in 0..3 {
                self.draw_line(color, &vs[i], &vs[(i + 1) % 3]);
            }
        } else {
            let [v0, v1, v2] = vs;
            if ((v2 - v0) ^ (v1 - v0)).z() > 0.0 {
                let (width, height) = self.texture.dimensions();
                let xs = vs.map(|v| v.x() as i32);
                let ys = vs.map(|v| v.y() as i32);
                let x_min = (*xs.iter().min().unwrap()).max(0);
                let y_min = (*ys.iter().min().unwrap()).max(0);
                let x_max = (*xs.iter().max().unwrap()).min(width as i32 - 1);
                let y_max = (*ys.iter().max().unwrap()).min(height as i32 - 1);

                let varying_z = &Vect::from(vs.map(|v| v.z()));

                for x in x_min..=x_max {
                    for y in y_min..=y_max {
                        let bc = barycentric(&Vect::from([x as f32, y as f32]), &vs);

                        if bc.x() >= 0.0 && bc.y() >= 0.0 && bc.z() >= 0.0 {
                            let bc = &Vect::from([bc.x() / screen[0].t(), bc.y() / screen[1].t(), bc.z() / screen[2].t()]);
                            let bc = bc / bc.sum();

                            let index = (y * width as i32 + x) as usize;
                            let z = varying_z * &bc;
                            if self.z_buffer[index] < z {
                                self.z_buffer[index] = z;
                                self.texture.set(x as usize, y as usize, shader.fragment(&bc));
                            }
                        }
                    }
                }

                // if settings.tangent_normals {
                //     let basis = &shader.darboux(&vec3(1.0, 0.0, 0.0)) * 10.0;
                //     self.draw_line(RED, v0, &(v0 + &basis.get_col(0)));
                //     self.draw_line(GREEN, v0, &(v0 + &basis.get_col(1)));
                //     self.draw_line(BLUE, v0, &(v0 + &basis.get_col(2)));
                // }
            }
        }
    }
}

impl Canvas<Color> {
    pub fn to_rgb8(&self) -> RgbImage {
        self.texture.to_rgb8()
    }

    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> EGLResult<()> {
        self.to_rgb8().save(path).map_err(to_egle(format!("Save image '{}'", path).as_str()))
    }
}

impl Canvas<f32> {
    pub fn to_rgb8(&self) -> RgbImage {
        self.texture.to_rgb8()
    }
}


fn barycentric(p: &Vect<2>, ps: &[Vect<3>; 3]) -> Vect<3> {
    let v1 = Vect::from([ps[2].x() - ps[0].x(), ps[1].x() - ps[0].x(), ps[0].x() - p.x()]);
    let v2 = Vect::from([ps[2].y() - ps[0].y(), ps[1].y() - ps[0].y(), ps[0].y() - p.y()]);
    let u = v1 ^ v2;

    if u.z().abs() < 0.5 {
        Vect::from([-1.0, -1.0, -1.0])
    } else {
        Vect::from([1.0 - (u.x() + u.y()) / u.z(), u.y() / u.z(), u.x() / u.z()])
    }
}

fn draw_line_helper(u1: i32, v1: i32, u2: i32, v2: i32, max_u: usize, max_v: usize, mut set: impl FnMut(usize, usize)) {
    if u1 > u2 {
        draw_line_helper(u2, v2, u1, v1, max_u, max_v, set);
        return;
    }

    let du = u2 - u1;
    let d_error = 2 * (v2 - v1).abs();
    let mut error = 0;
    let dv = if v1 < v2 { 1 } else { -1 };
    let mut v = v1;
    for u in u1..u2 + 1 {
        if 0 <= u && u < max_u as i32 && 0 <= v && v < max_v as i32 {
            set(u as usize, v as usize);
        }
        error += d_error;
        if error > du {
            v += dv;
            error -= 2 * du;
        }
    }
}
