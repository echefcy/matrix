use std::{cmp, fmt, ops};

#[derive(Debug, PartialEq)]
pub struct MatrixError;

#[derive(Debug, PartialEq)]
pub struct Matrix<T>
where
    T: fmt::Display,
{
    // `data` is a matrix with `rsize` rows and `csize` columns
    // RI: `csize` > 0, `rsize` > 0,`data.len() == rsize*csize`
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
        if !(rsize > 0 && csize > 0 && rsize * csize == data.len()) {
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
        for r in 0..cmp::min(10, self.rsize) {
            for c in 0..cmp::min(10, self.csize) {
                ret.push_str(&format!("{} ", self.data[r * self.csize + c]));
            }
            ret.push_str("\n");
        }
        ret.push_str(&format!("({} rows, {} columns)", self.rsize, self.csize));
        write!(f, "{}", ret)
    }
}

impl<T> ops::Add for Matrix<T> where T: fmt::Display {}
