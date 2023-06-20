use std::fmt::{Debug, Formatter};
use std::ops::{Add, Sub, BitXor, Mul, Div, MulAssign, DivAssign};


pub const fn vec3(x: f32, y: f32, z: f32) -> Vect<3> {
    Vect([x, y, z])
}


#[derive(Copy, Clone)]
pub struct Vect<const L: usize>([f32; L]);

impl<const L: usize> Debug for Vect<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vect{:?}", self.0)
    }
}

impl<const L: usize> Vect<L> {
    #[inline(always)]
    pub fn extend<const M: usize>(&self, fill: f32) -> Vect<M> {
        assert!(M > L);
        let mut data = [fill; M];
        data[..L].clone_from_slice(&self.0);
        Vect(data)
    }

    #[inline(always)]
    pub fn proj<const M: usize>(&self) -> Vect<M> {
        assert!(M < L);
        let mut data = [0.0; M];
        data.clone_from_slice(&self.0[..M]);
        Vect(data)
    }

    #[inline(always)]
    pub fn normalize(&self) -> Vect<L> {
        self / self.length()
    }

    #[inline(always)]
    pub fn length(&self) -> f32 {
        (self * self).sqrt()
    }

    pub fn sum(&self) -> f32 {
        let mut result = 0.0;
        for value in self.0 {
            result += value
        }
        result
    }

    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.at(0)
    }

    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.at(1)
    }

    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.at(2)
    }

    #[inline(always)]
    pub fn t(&self) -> f32 {
        self.at(3)
    }

    #[inline(always)]
    pub fn at(&self, index: usize) -> f32 {
        self.0[index]
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, value: f32) {
        self.0[index] = value;
    }

    #[inline(always)]
    pub fn add(&mut self, index: usize, value: f32) {
        self.0[index] += value;
    }
}

impl Vect<3> {
    pub fn to_point(&self) -> Vect<4> {
        self.extend(1.0)
    }

    pub fn to_vector(&self) -> Vect<4> {
        self.extend(0.0)
    }

    pub fn from_angles(angles: &Vect<3>) -> Vect<3> {
        let alpha = angles.x();
        let theta = angles.y();
        Vect::from([alpha.sin(), theta.sin(), alpha.cos() * theta.cos()])
    }

    pub const ZERO: Vect<3> = Vect([0.0, 0.0, 0.0]);
    pub const ONES: Vect<3> = Vect([1.0, 1.0, 1.0]);

    #[inline(always)]
    pub fn point(p: &Vect<4>) -> Vect<3> {
        Vect::from([p.0[0] / p.0[3], p.0[1] / p.0[3], p.0[2] / p.0[3]])
    }
}

impl<const L: usize> From<[f32; L]> for Vect<L> {
    #[inline(always)]
    fn from(data: [f32; L]) -> Self {
        Vect(data)
    }
}

impl<const L: usize> Add<&Vect<L>> for &Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn add(self, b: &Vect<L>) -> Self::Output {
        let mut data = [0.0f32; L];
        for i in 0..L {
            data[i] = self.0[i] + b.0[i];
        }
        Vect(data)
    }
}

impl<const L: usize> Add<Vect<L>> for Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn add(self, b: Vect<L>) -> Self::Output {
        &self + &b
    }
}


impl<const L: usize> Sub<&Vect<L>> for &Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn sub(self, b: &Vect<L>) -> Self::Output {
        let mut data = [0.0f32; L];
        for i in 0..L {
            data[i] = self.0[i] - b.0[i];
        }
        Vect(data)
    }
}

impl<const L: usize> Sub<Vect<L>> for Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn sub(self, b: Vect<L>) -> Self::Output {
        &self - &b
    }
}


impl<const L: usize> Mul<f32> for &Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn mul(self, b: f32) -> Self::Output {
        let mut data = [0.0f32; L];
        for i in 0..L {
            data[i] = self.0[i] * b;
        }
        Vect(data)
    }
}

impl<const L: usize> Mul<f32> for Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn mul(self, b: f32) -> Self::Output {
        &self * b
    }
}


impl<const L: usize> Div<f32> for &Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn div(self, b: f32) -> Self::Output {
        self * (1.0 / b)
    }
}

impl<const L: usize> Div<f32> for Vect<L> {
    type Output = Vect<L>;

    #[inline(always)]
    fn div(self, b: f32) -> Self::Output {
        &self / b
    }
}


impl<const L: usize> Mul<&Vect<L>> for &Vect<L> {
    type Output = f32;

    #[inline(always)]
    fn mul(self, b: &Vect<L>) -> Self::Output {
        let mut result = 0.0;
        for i in 0..L {
            result += self.0[i] * b.0[i];
        }
        result
    }
}

impl<const L: usize> Mul<Vect<L>> for Vect<L> {
    type Output = f32;

    #[inline(always)]
    fn mul(self, b: Vect<L>) -> Self::Output {
        &self * &b
    }
}


impl<const L: usize> MulAssign<&Matr<L, L>> for Vect<L> {
    #[inline(always)]
    fn mul_assign(&mut self, matrix: &Matr<L, L>) {
        let v: &Vect<L> = self;
        self.0 = (matrix * v).0;
    }
}


impl<const L: usize> DivAssign<&Matr<L, L>> for Vect<L> {
    #[inline(always)]
    fn div_assign(&mut self, matrix: &Matr<L, L>) {
        let v: &Vect<L> = self;
        self.0 = (&matrix.inverse() * v).0;
    }
}


impl BitXor<&Vect<3>> for &Vect<3> {
    type Output = Vect<3>;

    fn bitxor(self, b: &Vect<3>) -> Self::Output {
        Vect::from([
            self.y() * b.z() - b.y() * self.z(),
            self.z() * b.x() - b.z() * self.x(),
            self.x() * b.y() - b.x() * self.y(),
        ])
    }
}

impl BitXor<Vect<3>> for Vect<3> {
    type Output = Vect<3>;

    fn bitxor(self, b: Vect<3>) -> Self::Output {
        &self ^ &b
    }
}



#[derive(Copy, Clone, Debug)]
pub struct Matr<const R: usize, const C: usize>([[f32; C]; R]);

impl<const R: usize, const C: usize> Matr<R, C> {
    #[inline(always)]
    fn build(f: impl Fn(usize, usize) -> f32) -> Self {
        let mut data = [[0.0f32; C]; R];

        for i in 0..R {
            for j in 0..C {
                data[i][j] = f(i, j)
            }
        }

        Matr::from(data)
    }

    #[inline(always)]
    pub fn from_columns(value: [Vect<R>; C]) -> Self {
        Matr::build(|i, j| value[j].at(i))
    }

    pub fn same(value: f32) -> Self {
        Matr::from([[value; C]; R])
    }

    pub fn transpose(&self) -> Matr<C, R> {
        Matr::build(|i, j| self.0[j][i])
    }

    pub fn zero() -> Self {
        Matr::from([[0.0; C]; R])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.0[row][col] = value;
    }

    pub fn set_row(&mut self, row: usize, value: &Vect<C>) {
        self.0[row].clone_from_slice(&value.0);
    }

    pub fn get_row(&self, row: usize) -> Vect<C> {
        Vect::from(self.0[row])
    }

    pub fn set_col(&mut self, col: usize, value: &Vect<R>) {
        for row in 0..R {
            self.0[row][col] = value.0[row];
        }
    }

    pub fn get_col(&self, col: usize) -> Vect<R> {
        let mut data = [0.0; R];
        for row in 0..R {
            data[row] = self.0[row][col];
        }
        Vect(data)
    }

    pub fn proj<const R1: usize, const C1: usize>(&self) -> Matr<R1, C1> {
        assert!(R1 < R && C1 < C);
        let mut data = [[0.0; C1]; R1];
        for i in 0..R1 {
            data[i].clone_from_slice(&self.0[i][..R1]);
        }
        Matr::from(data)
    }
}

impl<const S: usize> Matr<S, S> {
    pub fn identity() -> Matr<S, S> {
        Matr::build(|i, j| if i == j { 1.0 } else { 0.0 })
    }

    pub fn inverse_transpose(&self) -> Matr<S, S> {
        self.inverse().transpose()
    }

    pub fn inverse(&self) -> Matr<S, S> {
        let mut mat = self.0;
        let mut inv = [[0.0; S]; S];
        for i in 0..S {
            inv[i][i] = 1.0;
        }

        for i in 0..S {
            let mut p = i;
            for r in i + 1..S {
                if mat[r][i].abs() > mat[p][i].abs() {
                    p = r;
                }
            }
            for r in 0..S {
                Self::inverse_sub((mat[r][i] - 1.0) / mat[p][i], &mut mat, &mut inv, p, r);
            }
            for r in 0..S {
                if i != r {
                    Self::inverse_sub(mat[r][i] / mat[i][i], &mut mat, &mut inv, i, r);
                }
            }
        }

        Matr::from(inv)
    }

    fn inverse_sub(q: f32, mat: &mut [[f32; S]; S], inv: &mut [[f32; S]; S], sr: usize, tr: usize) {
        for c in 0..S {
            mat[tr][c] -= q * mat[sr][c];
            inv[tr][c] -= q * inv[sr][c];
        }
    }

    pub fn rotate(axis1: usize, axis2: usize, angle: f32) -> Matr<S, S> {
        let mut matr = Matr::identity();
        let (sin, cos) = angle.sin_cos();
        matr.0[axis1][axis1] = cos;
        matr.0[axis1][axis2] = sin;
        matr.0[axis2][axis1] = -sin;
        matr.0[axis2][axis2] = cos;
        matr
    }
}

impl<const R: usize, const C: usize> From<[Vect<C>; R]> for Matr<R, C> {
    fn from(value: [Vect<C>; R]) -> Self {
        Matr::build(|i, j| value[i].at(j))
    }
}

impl<const R: usize, const C: usize> From<&Vec<Vec<f32>>> for Matr<R, C> {
    fn from(value: &Vec<Vec<f32>>) -> Self {
        Matr::build(|i, j| value[i][j])
    }
}

impl<const R: usize, const C: usize> From<[[f32; C]; R]> for Matr<R, C> {
    #[inline(always)]
    fn from(data: [[f32; C]; R]) -> Self {
        Matr(data)
    }
}

impl<const R: usize, const C: usize> Add<&Matr<R, C>> for &Matr<R, C> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn add(self, b: &Matr<R, C>) -> Self::Output {
        Matr::build(|i, j| self.0[i][j] + b.0[i][j])
    }
}


impl<const R: usize, const C: usize> Sub<&Matr<R, C>> for &Matr<R, C> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn sub(self, b: &Matr<R, C>) -> Self::Output {
        Matr::build(|i, j| self.0[i][j] - b.0[i][j])
    }
}


impl<const R: usize, const C: usize> Sub<Matr<R, C>> for Matr<R, C> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn sub(self, b: Matr<R, C>) -> Self::Output {
        &self - &b
    }
}

impl<const R: usize, const M: usize, const C: usize> Mul<&Matr<M, C>> for &Matr<R, M> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn mul(self, b: &Matr<M, C>) -> Self::Output {
        let mut data = [[0.0; C]; R];
        for i in 0..R {
            for j in 0..C {
                for k in 0..M {
                    data[i][j] += self.0[i][k] * b.0[k][j];
                }
            }
        }
        Matr::from(data)
    }
}

impl<const R: usize, const M: usize, const C: usize> Mul<Matr<M, C>> for Matr<R, M> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn mul(self, b: Matr<M, C>) -> Self::Output {
        &self * &b
    }
}


impl<const R: usize, const C: usize> Mul<&Vect<C>> for &Matr<R, C> {
    type Output = Vect<R>;

    #[inline(always)]
    fn mul(self, b: &Vect<C>) -> Self::Output {
        let mut data = [0.0; R];
        for i in 0..R {
            for j in 0..C {
                data[i] += self.0[i][j] * b.0[j];
            }
        }
        Vect(data)
    }
}

impl<const R: usize, const C: usize> Mul<Vect<C>> for Matr<R, C> {
    type Output = Vect<R>;

    #[inline(always)]
    fn mul(self, b: Vect<C>) -> Self::Output {
        &self * &b
    }
}



impl<const R: usize, const C: usize> Mul<&Matr<R, C>> for &Vect<R> {
    type Output = Vect<C>;

    #[inline(always)]
    fn mul(self, b: &Matr<R, C>) -> Self::Output {
        let mut data = [0.0; C];
        for c in 0..C {
            for r in 0..R {
                data[c] += self.0[r] * b.0[r][c];
            }
        }
        Vect(data)
    }
}

impl<const R: usize, const C: usize> Mul<Matr<R, C>> for Vect<R> {
    type Output = Vect<C>;

    #[inline(always)]
    fn mul(self, b: Matr<R, C>) -> Self::Output {
        &self * &b
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &Matr<R, C> {
    type Output = Matr<R, C>;

    #[inline(always)]
    fn mul(self, b: f32) -> Self::Output {
        Matr::build(|i, j| self.0[i][j] * b)
    }
}

impl<const S: usize> MulAssign<&Matr<S, S>> for Matr<S, S> {
    fn mul_assign(&mut self, matrix: &Matr<S, S>) {
        self.0 = (matrix * &*self).0;
    }
}

impl<const S: usize> DivAssign<&Matr<S, S>> for Matr<S, S> {
    fn div_assign(&mut self, matrix: &Matr<S, S>) {
        self.0 = (&matrix.inverse() * &*self).0;
    }
}

#[cfg(test)]
mod tests {
    use crate::linear::Matr;

    #[test]
    fn test_inverse() {
        test_inverse_matrix(Matr::from([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]));
        test_inverse_matrix(Matr::from([[0.0, -1.0, 2.0], [2.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]));
        test_inverse_matrix(Matr::from([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]));
        test_inverse_matrix(Matr::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
    }

    fn test_inverse_matrix(matrix: Matr<3, 3>) {
        let inverse = matrix.inverse();
        let zero = matrix * inverse - Matr::identity();
        println!("mat {:?}", matrix);
        println!("inv {:?}", inverse);
        println!("zer {:?}", zero);
        for i in 0..zero.0.len() {
            for j in 0..zero.0[i].len() {
                assert!(zero.0[i][j].abs() < 1e-6, "{} {}", i, j);
            }
        }
    }
}
