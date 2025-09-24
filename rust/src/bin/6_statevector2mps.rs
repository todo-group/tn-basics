use anyhow::Result;
use ndarray::{arr1, Array1, Array2, ArrayD, Axis, Ix2};
use ndarray_linalg::svd::SVD; // A = U * diag(S) * Vt

fn truncate_svd(u: &Array2<f64>, s: &Array1<f64>, vt: &Array2<f64>, cutoff: f64)
    -> (Array2<f64>, Array2<f64>, usize)
{
    let r = s.iter().take_while(|&&x| x > cutoff * s[0]).count();
    let u2  = u.slice(s![.., 0..r]).to_owned();        // (m, r)
    let s2  = s.slice(s![0..r]).to_owned();            // (r)
    let vt2 = vt.slice(s![0..r, ..]).to_owned();       // (r, n)

    // v = diag(S) * Vt  -> (r, r) * (r, n) = (r, n)
    let mut diag = Array2::<f64>::zeros((r, r));
    for i in 0..r { diag[(i,i)] = s2[i]; }
    let v = diag.dot(&vt2);
    (u2, v, r)
}

fn main() -> Result<()> {
    let cutoff = 1e-10_f64;

    // Bell state
    println!("Bell state:");
    let v = arr1(&[1.0, 0.0, 0.0, 1.0]) / 2f64.sqrt();
    let v_mat = v.into_shape((2, 2))?;  // (2,2)
    let (u_opt, s, vt_opt) = v_mat.svd(true, true)?;
    let (u, vt) = (u_opt.unwrap(), vt_opt.unwrap());
    println!("singular values: {:?}", s);
    let (u_l, v_r, _r) = truncate_svd(&u, &s, &vt, cutoff);
    println!("tensors [{:?}, {:?}])\n", u_l, v_r);

    // GHZ state
    let n: usize = 6;
    println!("n={} GHZ state:", n);
    let mut v = Array1::<f64>::zeros(1usize << n);
    v[0] = 1.0 / 2f64.sqrt();
    v[v.len()-1] = 1.0 / 2f64.sqrt();

    let mut mps: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;

    // V は都度 (rank*2, -1) に整形してから SVD
    let mut V2 = v.into_shape((rank*2, (1usize<<n)/(rank*2)))?.to_owned();

    for i in 0..(n-1) {
        // 形状は常に (rank*2, cols)
        let (u_opt, s, vt_opt) = V2.svd(true, true)?;
        let (u, vt) = (u_opt.unwrap(), vt_opt.unwrap());
        println!("{}: singular values: {:?}", i, s);

        let (mut u_l, v_r, r_new) = truncate_svd(&u, &s, &vt, cutoff);

        if i > 0 {
            // u_l: (rank*2, r_new) -> (rank, 2, r_new)
            u_l = u_l.into_shape((rank, 2, r_new))?;
            mps.push(u_l.into_dyn());
        } else {
            // 最初だけ (2, r_new)
            mps.push(u_l.into_dyn());
        }

        V2 = v_r;                 // 次の「左正準右端」へ
        rank = r_new;

        if i < n-2 {
            // 次の分解に向け (rank*2, -1) へ整形
            let cols = V2.len() / (rank * 2);
            V2 = V2.into_shape((rank * 2, cols))?;
        }
    }

    // 最後に (rank, 2) に整形して右端テンソル
    let last = V2.into_shape((rank, 2))?;
    mps.push(last.into_dyn());

    println!("tensors: {:?}\n", mps);
    Ok(())
}
