use ndarray::Array;
use ndarray_einsum::einsum;
use rand::random;
use tn_basics::MapStrToAnyhowErr;

// Tensor contraction examples

fn main() -> anyhow::Result<()> {
    println!("matrix-matrix multiplication");
    let a = Array::from_shape_fn((2, 3), |_| random::<f64>());
    let b = Array::from_shape_fn((3, 4), |_| random::<f64>());
    println!("A: shape {:?}\n{}\n", a.shape(), a);
    println!("B: shape {:?}\n{}\n", b.shape(), b);
    println!("contract: A, B -> C");
    let c = einsum("ij,jk->ik", &[&a, &b]).map_str_err()?;
    println!("C: shape {:?}\n{}\n", c.shape(), c);

    println!("more complex contraction");
    let a = Array::from_shape_fn((2, 3, 4, 5), |_| random::<f64>());
    let b = Array::from_shape_fn((4, 3), |_| random::<f64>());
    let c = Array::from_shape_fn((5, 3, 4), |_| random::<f64>());
    println!("A: shape {:?}\n{}\n", a.shape(), a);
    println!("B: shape {:?}\n{}\n", b.shape(), b);
    println!("C: shape {:?}\n{}\n", c.shape(), c);
    println!("contract: A, B, C -> D");
    let d = einsum("ijlm,ln,mnk->ijk", &[&a, &b, &c]).map_str_err()?;
    println!("D: shape {:?}\n{}\n", d.shape(), d);
    Ok(())
}
