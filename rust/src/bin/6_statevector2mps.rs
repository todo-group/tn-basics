use ndarray::{Array1, Array2, ArrayD, arr1, s};
use ndarray_linalg::Norm;
use tn_basics::EasySVD; // A = U * diag(S) * Vt

fn main() -> anyhow::Result<()> {
    let cutoff = 1e-10_f64;

    // Bell state
    println!("Bell state:");
    let v = arr1(&[1.0, 0.0, 0.0, 1.0]) / 2f64.sqrt();
    let v = v.to_shape((2, 2))?;
    let (u, s, vt) = v.thin_svd()?;
    println!("singular values: {}", s);

    let rank_new = s
        .iter()
        .position(|&x| x <= cutoff * s[0])
        .unwrap_or(s.len());
    let u = u.slice_move(s![.., 0..rank_new]);
    let s = s.slice_move(s![0..rank_new]);
    let vt = vt.slice_move(s![0..rank_new, ..]);
    let v = Array2::from_diag(&s).dot(&vt);
    println!("tensors [{}, {}])\n", u, v);

    // GHZ state
    let n: usize = 16;
    println!("n={} GHZ state:", n);
    let mut v = Array1::<f64>::zeros(1usize << n);
    let v_len = v.len();
    v[0] = 1.0 / 2f64.sqrt();
    v[v_len - 1] = 1.0 / 2f64.sqrt();
    let mut v = v.into_dyn();

    let mut mps: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;

    for i in 0..(n - 1) {
        let v_view = v.to_shape((rank * 2, v.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd()?;
        println!("{}: singular values: {}", i, s);

        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice_move(s![.., 0..rank_new]);
        let s = s.slice_move(s![0..rank_new]);
        let vt = vt.slice_move(s![0..rank_new, ..]);
        let u = if i > 0 {
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_dyn()
        };
        mps.push(u);
        v = Array2::from_diag(&s).dot(&vt).into_dyn();
        rank = rank_new;
    }
    let v = v.to_shape((rank, 2))?.into_owned().into_dyn();
    mps.push(v);

    println!("tensors:");
    for (i, t) in mps.iter().enumerate() {
        println!("  {}:\n{}", i, t);
    }

    // random state
    let n: usize = 16;
    println!("n={} random state:", n);
    let mut v = Array1::<f64>::from_shape_fn(1usize << n, |_| rand::random::<f64>());
    v /= v.norm();
    let mut v = v.into_dyn();

    let mut mps: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;

    for i in 0..(n - 1) {
        let v_view = v.to_shape((rank * 2, v.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd()?;
        println!("{}: singular values: {}", i, s);

        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice_move(s![.., 0..rank_new]);
        let s = s.slice_move(s![0..rank_new]);
        let vt = vt.slice_move(s![0..rank_new, ..]);
        let u = if i > 0 {
            // u_l: (rank*2, r_new) -> (rank, 2, r_new)
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_dyn()
        };
        mps.push(u);
        v = Array2::from_diag(&s).dot(&vt).into_dyn();
        rank = rank_new;
    }
    let v = v.to_shape((rank, 2))?.into_owned().into_dyn();
    mps.push(v);

    println!("virtual bond dimensions:");
    for i in 1..n {
        println!("{}:: {}", i, mps[i].shape()[0]);
    }

    Ok(())
}
