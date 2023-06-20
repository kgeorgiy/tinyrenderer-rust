use std::fs;
use std::time::Instant;
use image::RgbImage;

use show_image::{create_window, ImageInfo, ImageView, WindowOptions, WindowProxy};
use show_image::event::{VirtualKeyCode, WindowEvent};
use crate::canvas::{Canvas, CanvasSettings};
use crate::depth_shader::DepthShader;

use crate::egl::{Color, EGLError, EGLResult, rgb_color, Shader, to_egle};
use crate::linear::{Matr, vec3, Vect};
use crate::render_shader::{Light, Normals, ShaderSettings, UniversalShader};
use crate::texture::Texture;

mod linear;
mod egl;
mod texture;
mod canvas;
mod render_shader;
mod depth_shader;

const WHITE: Color = rgb_color(1.0, 1.0, 1.0);
const GREY: Color = rgb_color(0.1, 0.1, 0.1);
const BLUE: Color = rgb_color(0.0, 0.0, 1.0);
const GREEN: Color = rgb_color(0.5, 1.0, 0.5);
const RED: Color = rgb_color(1.0, 0.0, 0.0);
const PI: f32 = std::f32::consts::PI;

fn run() -> EGLResult<()> {
    let size = 450 * 2;

    let start = Instant::now();

    let directory = "resources/african_head";
    // let directory = "resources/diablo3_pose";
    // let directory = "resources/chess";
    let mut scene = Scene {
        model: Model::load(path(directory, "model.obj"))?,
        view_matrix: Matr::identity(),
        light_matrix: Matr::identity(),
        scale: 0.8,
    };
    let light_direction = &Matr::rotate(0, 2, PI / 2.0) * &Vect::from([0.0, 0.0, 1.0]);
    let specular = Texture::load_specular(path(directory, "specular.png"))?;
    let shader_settings = ShaderSettings {
        // texture: Texture::load_texture(path(directory, "../grid.png"))?,
        texture: Texture::load_texture(path(directory, "diffuse.png"))?,

        // normals: Normals::Flat,
        // normals: Normals::Phong,
        normals: Normals::Global(Texture::load_normals(path(directory, "nm.png"))?),
        // normals: Normals::Tangent(Texture::load_normals(path(directory, "nm_tangent.png"))?),

        // solid_color: Some(WHITE),
        solid_color: None,

        cell_shading: false,
        // light: Light::Specular(light_direction, &self.shader_settings.specular),
        light: Light::Composite(5.0, vec![
            (Light::Ambient, 1.0),
            (Light::Diffuse(light_direction), 3.0),
            (Light::Specular(light_direction, &specular), 1.0),
        ]),
    };

    let mut settings = CanvasSettings::new();
    // settings.axes = Some((RED, GREEN, BLUE));
    // settings.wireframe = Some(WHITE);
    // settings.normals = Some(BLUE);
    settings.tangent_normals = Some((RED, GREEN, BLUE));

    // let canvas = &mut Canvas::new(size, size, GREY, settings);
    // let depth = &mut Canvas::new(size, size, -f32::INFINITY, CanvasSettings::new());

    show(&mut View {
        scene: &mut scene,
        canvas: Canvas::new(size, size, GREY, settings),
        shader: &mut UniversalShader::new(&shader_settings),
        to_rgb8: |c| c.to_rgb8(),
    })?;
    // show(&mut View {
    //     scene: &mut scene,
    //     canvas: Canvas::new(size, size, -f32::INFINITY, CanvasSettings::new()),
    //     shader: &mut DepthShader::new(&light_direction),
    //     to_rgb8: |c| c.to_rgb8(),
    // })?;
    // for i in 0..16 {
    //     println!("Render {}", i);
    //     scene.angle = i as f32 * PI / 8.0;
    //
    //     scene.display(&window, canvas)?;
    //     canvas.save(format!("__output/test-{:02}.tga", i).as_str())?;
    // }
    // gui::run(1024, 1024, &|img: &mut RgbImage| {
    // })
    Ok(println!("done in {}ms", start.elapsed().as_millis()))
}

fn show<T: Copy>(view: &mut View<T>) -> Result<(), EGLError> {
    let d_angle = PI / 16.0;
    let rot_y = &Matr::rotate(0, 2, d_angle);
    let rot_x = &Matr::rotate(1, 2, d_angle);
    let rot_z = &Matr::rotate(0, 1, d_angle);
    let window_options: WindowOptions = WindowOptions::default()
        .set_fullscreen(true);
    let window = create_window("image", window_options).map_err(to_egle("Create window"))?;


    view.display(&window)?;

    for event in window.event_channel().map_err(to_egle("Event channel"))? {
        if let WindowEvent::KeyboardInput(event) = event {
            let input = event.input;
            if input.state.is_pressed() {
                if let Some(key) = input.key_code {
                    let mut redraw = true;
                    match key {
                        VirtualKeyCode::Escape => break,
                        VirtualKeyCode::Up => view.scene.light_matrix *= rot_x,
                        VirtualKeyCode::Down => view.scene.light_matrix /= rot_x,
                        VirtualKeyCode::Right => view.scene.light_matrix *= rot_y,
                        VirtualKeyCode::Left => view.scene.light_matrix /= rot_y,
                        VirtualKeyCode::S => view.scene.view_matrix *= rot_x,
                        VirtualKeyCode::W => view.scene.view_matrix /= rot_x,
                        VirtualKeyCode::A => view.scene.view_matrix *= rot_y,
                        VirtualKeyCode::D => view.scene.view_matrix /= rot_y,
                        VirtualKeyCode::Q => view.scene.view_matrix *= rot_z,
                        VirtualKeyCode::E => view.scene.view_matrix /= rot_z,
                        other => {
                            println!("Key pressed: {:?}", other);
                            redraw = false;
                        },
                    }
                    if redraw {
                        view.display(&window)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn path(directory: &str, file: &str) -> String {
    format!("{}/{}", directory, file)
}


struct Scene {
    model: Model,
    scale: f32,
    view_matrix: Matr<3, 3>,
    light_matrix: Matr<3, 3>,
}

impl Scene {
    fn render<T: Copy>(&self, mut canvas: &mut Canvas<T>, shader: &mut dyn Shader<T>) {
        let start = Instant::now();

        let projection = &egl::projection(-5.0);
        let lookat = &egl::look_at(
            &(&self.view_matrix * &Vect::from_angles(&vec3(PI + PI / 8.0, -PI / 16.0, 0.0))),
            &vec3(0.0, 0.0, 0.0),
            &(&self.view_matrix * &vec3(0.0, 1.0, 0.0)),
        );
        let transform = &(projection * lookat);
        let viewport = &canvas.viewport(self.scale);

        shader.set_transforms(transform, &self.light_matrix);
        self.model.render(&mut canvas, &(viewport * transform), shader);

        println!("Display in {}ms", start.elapsed().as_millis());
    }
}

#[show_image::main]
fn main() {
    if let Err(error) = run() {
        println!("Error: {}", error.message)
    }
}

struct Model {
    vertices: Vec<Vect<3>>,
    faces: Vec<Face>,
    texture_vertices: Vec<Vect<3>>,
    vertex_normals: Vec<Vect<3>>,
}

impl Model {
    pub fn load(obj_file: String) -> EGLResult<Model> {
        let mut vertices: Vec<Vect<3>> = Vec::new();
        let mut faces: Vec<Face> = Vec::new();
        let mut texture_vertices: Vec<Vect<3>> = Vec::new();
        let mut vertex_normals: Vec<Vect<3>> = Vec::new();
        fs::read_to_string(obj_file).map_err(to_egle("Open model file"))?
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with("#"))
            .map(|line| -> Vec<&str> { line.split_whitespace().collect() })
            .for_each(|line| {
                if line[0] == "v" {
                    vertices.push(parse_vec3(&line))
                } else if line[0] == "f" {
                    faces.push(Face {
                        angles: [
                            parse_angle(0, line[1]),
                            parse_angle(1, line[2]),
                            parse_angle(2, line[3]),
                        ]
                    })
                } else if line[0] == "vt" {
                    texture_vertices.push(parse_vec3(&line));
                } else if line[0] == "vn" {
                    vertex_normals.push(parse_vec3(&line).normalize())
                } else if line[0] == "g" || line[0] == "s" {} else {
                    println!("{}", line[0]);
                    assert!(false);
                }
            });
        Ok(Model { vertices, faces, texture_vertices, vertex_normals })
    }

    fn render<T: Copy>(&self, canvas: &mut Canvas<T>, transform: &Matr<4, 4>, shader: &mut dyn Shader<T>) {
        canvas.clear();

        for face in &self.faces {
            let screen: [Vect<4>; 3] = core::array::from_fn(|index| {
                let angle = &face.angles[index];
                let vertex = &self.vertices[angle.vertex_index];
                let normal = self.vertex_normals[angle.normal_index];
                let screen = transform * &vertex.extend(1.0);
                shader.vertex(angle.index, vertex, &normal, &self.texture_vertices[angle.texture_index], &screen);
                screen
            });

            canvas.triangle(screen, shader);
        }

        if let Some(color) = canvas.settings.normals {
            for i in 0..self.vertices.len() {
                let vertex = &self.vertices[i];
                let true_normal = &(&self.vertex_normals[i] * 0.05);
                canvas.draw_line_transformed(color, &transform, &vertex, &(vertex + true_normal));

                // let proj_normal = &(&shader.transform_normal(&self.vertex_normals[i]).normalize() * 10.0);
                // let projected = &Vect::point(&(transform * &vertex.to_point()));
                // canvas.draw_line(GREEN, projected, &(projected + proj_normal));
            }
        }
        if let Some((x, y, z)) = canvas.settings.axes {
            let origin = Vect::ZERO;
            canvas.draw_line_transformed(x, &transform, &origin, &vec3(1.0, 0.0, 0.0));
            canvas.draw_line_transformed(y, &transform, &origin, &vec3(0.0, 1.0, 0.0));
            canvas.draw_line_transformed(z, &transform, &origin, &vec3(0.0, 0.0, 1.0));
        }
    }
}

struct Angle {
    index: usize,
    vertex_index: usize,
    texture_index: usize,
    normal_index: usize,
}

struct Face {
    angles: [Angle; 3],
}

fn parse_vec3(line: &Vec<&str>) -> Vect<3> {
    Vect::from([
        parse_float(line[1]),
        parse_float(line[2]),
        parse_float(line[3]),
    ])
}

fn parse_angle(index: usize, angle: &str) -> Angle {
    let parts: Vec<&str> = angle.split("/").collect();
    return Angle {
        index,
        vertex_index: parse_index(parts[0]) - 1,
        texture_index: parse_index(parts[1]) - 1,
        normal_index: parse_index(parts[2]) - 1,
    };
}

fn parse_float(string: &str) -> f32 {
    string.parse().expect("float")
}

fn parse_index(string: &str) -> usize {
    string.parse().expect("usize")
}


struct View<'t, T> {
    canvas: Canvas<T>,
    scene: &'t mut Scene,
    shader: &'t mut dyn Shader<T>,
    to_rgb8: fn(&Canvas<T>) -> RgbImage,
}

impl<'t, T: Copy> View<'t, T> {
    fn display(&mut self, window: &WindowProxy) -> Result<(), EGLError> {
        self.scene.render(&mut self.canvas, self.shader);
        let to_rgb8 = self.to_rgb8;
        let pixels = &(to_rgb8(&self.canvas).into_raw()[..]);
        let view = ImageView::new(
            ImageInfo::rgb8(self.canvas.width() as u32, self.canvas.height() as u32),
            pixels
        );
        window.set_image("", view).map_err(to_egle("Set image"))
    }
}
