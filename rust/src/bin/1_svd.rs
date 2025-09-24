use ndarray::prelude::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::OperationNorm;
use tn_basics::ThinSVD;

extern crate blas_src;

fn main() -> anyhow::Result<()> {
    let a: Array2<f64> = array![
        [1., 2., 3.],
        [6., 4., 5.],
        [8., 9., 7.],
        [10., 11., 12.]
    ];
    println!("A =\n{}\n", a);

    // (thin) SVD
    let (u, s, vt) = a.thin_svd(true, true)?;
    let u = u.unwrap();
    let vt = vt.unwrap();
    println!("U =\n{}\n", u);
    println!("S =\n{}\n", s);
    println!("Vt =\n{}\n", vt);

    // reconstruct A
    let s_mat = Array2::from_diag(&s);
    let a_reconstructed = u.dot(&s_mat).dot(&vt);
    println!("reconstructed A =\n{}\n", a_reconstructed);

    // full SVD
    let (u_full, s_full, vt_full) = a.svd(true, true)?;
    let u_full = u_full.unwrap();
    let vt_full = vt_full.unwrap();
    println!("U (full) =\n{}\n", u_full);
    println!("S (full) =\n{}\n", s_full);
    println!("Vt (full) =\n{}\n", vt_full);

    // reconstruct A (full SVD)
    let mut s_full_mat = Array2::<f64>::zeros((4, 3));
    for (i, &val) in s_full.iter().enumerate() {
        s_full_mat[(i, i)] = val;
    }
    let a_reconstructed_full = u_full.dot(&s_full_mat).dot(&vt_full);
    println!("reconstructed A =\n{}\n", a_reconstructed_full);

    // rank-2 approximation
    let r = 2;
    let ur = u_full.slice(s![.., 0..r]).to_owned();
    let sr = Array2::from_diag(&s_full.slice(s![0..r]).to_owned());
    let vr = vt_full.slice(s![0..r, ..]).to_owned();

    let a_rank2 = ur.dot(&sr).dot(&vr);
    println!("rank-2 approximation of A =\n{}\n", a_rank2);

    let diff = &a - &a_rank2;
    let err = diff.opnorm_fro()?;
    println!("Frobenius norm of the error = {}", err);

    Ok(())
}
