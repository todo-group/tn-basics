use ndarray::{Array, Ix2, Ix3};
use ndarray_einsum::einsum;
use rand::random;
use tn_basics::MapStrToAnyhowErr;

extern crate blas_src;

// Tensor contraction examples

fn main() -> anyhow::Result<()> {
    println!("matrix-matrix multiplication");
    let a = Array::from_shape_fn((2, 3), |_| random::<f64>());
    let b = Array::from_shape_fn((3, 4), |_| random::<f64>());
    println!("A: shape {:?}\n{}\n", a.dim(), a);
    println!("B: shape {:?}\n{}\n", b.dim(), b);
    println!("contract: A, B -> C");
    let c = einsum("ij,jk->ik", &[&a, &b])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    println!("C: shape {:?}\n{}\n", c.dim(), c);

    println!("more complex contraction");
    let a = Array::from_shape_fn((2, 3, 4, 5), |_| random::<f64>());
    let b = Array::from_shape_fn((4, 3), |_| random::<f64>());
    let c = Array::from_shape_fn((5, 3, 4), |_| random::<f64>());
    println!("A: shape {:?}\n{}\n", a.dim(), a);
    println!("B: shape {:?}\n{}\n", b.dim(), b);
    println!("C: shape {:?}\n{}\n", c.dim(), c);
    println!("contract: A, B, C -> D");
    let d = einsum("ijlm,ln,mnk->ijk", &[&a, &b, &c])
        .map_str_err()?
        .into_dimensionality::<Ix3>()?;
    println!("D: shape {:?}\n{}\n", d.dim(), d);
    Ok(())
}
