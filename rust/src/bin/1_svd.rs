use ndarray::{Array2, array, s};
use ndarray_linalg::OperationNorm;
use tn_basics::EasySVD;

extern crate blas_src;

// SVD and low-rank approximation of a matrix

fn main() -> anyhow::Result<()> {
    let a = array![[1., 2., 3.], [6., 4., 5.], [8., 9., 7.], [10., 11., 12.]];
    println!("A =\n{}\n", a);

    // (thin) SVD
    let (u, s, vt) = a.thin_svd()?;
    println!("U =\n{}\n", u);
    println!("S =\n{}\n", s);
    println!("Vt =\n{}\n", vt);

    // reconstruct A
    let s_matrix = Array2::from_diag(&s);
    let a_reconstructed = u.dot(&s_matrix).dot(&vt);
    println!("reconstructed A =\n{}\n", a_reconstructed);

    // full SVD
    let (u_full, s_full, vt_full) = a.full_svd()?;
    println!("U (full) =\n{}\n", u_full);
    println!("S (full) =\n{}\n", s_full);
    println!("Vt (full) =\n{}\n", vt_full);

    // reconstruct A (full SVD)
    let mut s_matrix = Array2::<f64>::zeros((u_full.ncols(), vt_full.nrows()));
    s_matrix.diag_mut().assign(&s_full);
    let s_matrix = s_matrix;
    let a_reconstructed_full = u_full.dot(&s_matrix).dot(&vt_full);
    println!("reconstructed A =\n{}\n", a_reconstructed_full);

    // rank-2 approximation
    let r = 2;
    let ur = u_full.slice(s![.., 0..r]);
    let sr = Array2::from_diag(&s_full.slice(s![0..r]));
    let vr = vt_full.slice(s![0..r, ..]);
    let a_rank2 = ur.dot(&sr).dot(&vr);

    println!("rank-2 approximation of A =\n{}\n", a_rank2);
    println!(
        "Frobenius norm of the error = {}",
        (&a - &a_rank2).opnorm_fro()?
    );

    Ok(())
}
