#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}


mod matrix{
    use std::ops::Mul;
    use std::ops::Add;
    struct Matrix<T>{
        rsize: usize,
        csize: usize,
        data: Vec<T>
    }
    impl<T> Matrix<T>{
        
    }
}