use ndarray::Array;
use ndarray_einsum::einsum;
use rand::random;

extern crate blas_src;

fn main() {
    println!("matrix-matrix multiplication");
    let a = Array::from_shape_fn((2, 3), |_| random::<f64>());
    let b = Array::from_shape_fn((3, 4), |_| random::<f64>());
    println!("A: shape {:?}\n{:?}\n", a.dim(), a);
    println!("B: shape {:?}\n{:?}\n", b.dim(), b);
    println!("contract: A, B -> C");
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    println!("C: shape {:?}\n{:?}\n", c.dim(), c);

    println!("more complex contraction");
    let a = Array::from_shape_fn((2, 3, 4, 5), |_| random::<f64>());
    let b = Array::from_shape_fn((4, 3), |_| random::<f64>());
    let c = Array::from_shape_fn((5, 3, 4), |_| random::<f64>());
    println!("A: shape {:?}", a.dim());
    println!("B: shape {:?}", b.dim());
    println!("C: shape {:?}", c.dim());
    println!("contract: A, B, C -> D");
    let d = einsum("ijlm,ln,mnk->ijk", &[&a, &b, &c]).unwrap();
    println!("D: shape {:?}\n{:?}\n", d.dim(), d);
}
