use std::{cmp, fmt, ops};

#[derive(Debug)]
pub struct MatrixError;

#[derive(Debug)]
pub struct Matrix<T>
where
    T: fmt::Display,
{
    rsize: usize,
    csize: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: fmt::Display,
{
    /// Constructs a new `Matrix` Result with `rsize` rows, `csize` columns, and elements `data`.
    /// Requires: dimensions match the length of `data`
    pub fn new(rsize: usize, csize: usize, data: Vec<T>) -> Result<Matrix<T>, MatrixError> {
        if rsize * csize != data.len() {
            Err(MatrixError)
        } else {
            Ok(Matrix { rsize, csize, data })
        }
    }
}

impl<T> fmt::Display for Matrix<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ret = String::new();
        for row in 0..cmp::min(10, self.rsize) {
            for col in 0..cmp::min(10, self.csize) {
                ret.push_str(&format!("{} ", self.data[row * self.csize + col]));
            }
            ret.push_str("\n");
        }
        ret.push_str(&format!("({} rows, {} columns)", self.rsize, self.csize));
        write!(f, "{}", ret)
    }
}

impl<T> ops::Add for Matrix<T> where T: fmt::Display {}
