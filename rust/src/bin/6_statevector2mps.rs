use ndarray::{Array1, Array2, ArrayD, array, s};
use ndarray_linalg::Norm;
use tn_basics::EasySVD;

// Generate MPS from statevector

fn main() -> anyhow::Result<()> {
    let cutoff = 1e-10_f64;

    // Bell state
    println!("Bell state:");
    let v = array![1.0, 0.0, 0.0, 1.0] / 2f64.sqrt();

    let v = v.into_shape_with_order((2, 2))?;
    let (u, s, vt) = v.thin_svd()?;
    println!("singular values: {s}");
    let rank_new = s
        .iter()
        .position(|&x| x <= cutoff * s[0])
        .unwrap_or(s.len());
    let u = u.slice(s![.., 0..rank_new]);
    let s = s.slice(s![0..rank_new]);
    let vt = vt.slice(s![0..rank_new, ..]);
    let v = Array2::from_diag(&s).dot(&vt);
    println!("tensors [{u}, {v}])\n");

    // GHZ state
    let n: usize = 16;
    println!("n={n} GHZ state:");
    let mut v = Array1::<f64>::zeros(1usize << n);
    let v_len = v.len();
    v[0] = 1.0 / 2f64.sqrt();
    v[v_len - 1] = 1.0 / 2f64.sqrt();
    let mut v = v.into_shape_with_order((1, v_len))?;
    let mut mps: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;
    for i in 0..(n - 1) {
        let v_view = v.to_shape((rank * 2, v.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd()?;
        println!("{i}: singular values: {s}");
        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice(s![.., 0..rank_new]);
        let s = s.slice(s![0..rank_new]);
        let vt = vt.slice(s![0..rank_new, ..]);
        let u = if i > 0 {
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_owned().into_dyn()
        };
        mps.push(u);
        v = Array2::from_diag(&s).dot(&vt);
        rank = rank_new;
    }
    let v = v.into_shape_clone((rank, 2))?.into_dyn();
    mps.push(v);
    println!("tensors:[");
    for t in &mps {
        println!("{t}");
    }
    println!("]");

    // random state
    let n: usize = 16;
    println!("n={n} random state:");
    let mut v = Array1::<f64>::from_shape_fn(1usize << n, |_| rand::random::<f64>());
    v /= v.norm();
    let v_len = v.len();
    let mut v = v.into_shape_with_order((1, v_len))?;

    let mut mps: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;
    for i in 0..(n - 1) {
        let v_view = v.to_shape((rank * 2, v.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd()?;
        println!("{i}: singular values: {s}");
        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice(s![.., 0..rank_new]);
        let s = s.slice(s![0..rank_new]);
        let vt = vt.slice(s![0..rank_new, ..]);
        let u = if i > 0 {
            // u_l: (rank*2, r_new) -> (rank, 2, r_new)
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_owned().into_dyn()
        };
        mps.push(u);
        v = Array2::from_diag(&s).dot(&vt);
        rank = rank_new;
    }
    let v = v.into_shape_clone((rank, 2))?.into_dyn();
    mps.push(v);
    println!("virtual bond dimensions:");
    for (i, ti) in mps[1..n].iter().enumerate() {
        println!("{i}: {}", ti.shape()[0]);
    }

    Ok(())
}
