use matrix::matrix;

fn main() {
    let matvec = vec![1.2, 2., 3., 4.4, 5., 6.];
    let mat = matrix::Matrix::new(3, 2, matvec).unwrap();
    println!("{}", mat);
}
