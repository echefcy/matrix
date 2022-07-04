#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

pub mod matrix;

#[cxx::bridge]
mod ffi{
    unsafe extern "C++"{
        include!("matrix/gpu/ops.h");
        
        fn sum(&CxxVector<float>, &CxxVector<float>) -> CxxVector<float>;
    }
}