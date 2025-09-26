use anyhow::Result;
use ndarray::{arr1, Array1, Array2, Array3, Axis};
use ndarray_einsum_beta::einsum;
use ndarray_linalg::svd::SVD;
use plotters::prelude::*;

fn truncate_svd(u: &Array2<f64>, s: &Array1<f64>, vt: &Array2<f64>, cutoff: f64, max_rank: usize)
    -> (Array2<f64>, Array2<f64>, usize)
{
    let mut r = 0usize;
    let thresh = cutoff * s[0];
    for &sv in s.iter() {
        if sv > thresh { r += 1; } else { break; }
        if r == max_rank { break; }
    }
    assert!(r > 0, "All singular values truncated");
    let u2  = u.slice(ndarray::s![.., 0..r]).to_owned();    // (m, r)
    let s2  = s.slice(ndarray::s![0..r]).to_owned();        // (r)
    let vt2 = vt.slice(ndarray::s![0..r, ..]).to_owned();   // (r, n)

    // diag(S) * Vt
    let mut diag = Array2::<f64>::zeros((r, r));
    for i in 0..r { diag[(i,i)] = s2[i]; }
    let v_next = diag.dot(&vt2);                            // (r, n)
    (u2, v_next, r)
}

fn main() -> Result<()> {
    let depth: usize   = 8;
    let npoints: usize = 1 << depth;
    let cutoff: f64    = 1e-10;
    let max_rank: usize = 4;

    // target function
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, npoints);
    let y: Array1<f64> = x.mapv(|t| t.exp());
    // let y = x.mapv(|t| t.sin());
    // let y = x.mapv(|t| ( -(((t-0.5)/0.1).powi(2)) ).exp() / (0.1 * std::f64::consts::PI.sqrt()) );

    if depth < 5 {
        println!("x = {:?}", x);
        println!("y = {:?}\n", y);
    }

    // QTT decomposition
    let mut yt = y.clone();          // (npoints,)
    let mut qtt: Vec<Array3<f64>> = Vec::new(); // 中間は (rank,2,rank_next) で持つ
    let mut rank: usize = 1;

    for k in 0..(depth-1) {
        println!("depth: {}", k);
        // reshape (rank*2, -1)
        let cols = yt.len() / (rank * 2);
        let yt_mat = yt.into_shape((rank*2, cols))?;

        let (u_opt, s, vt_opt) = yt_mat.svd(true, true)?;
        let (u, vt) = (u_opt.unwrap(), vt_opt.unwrap());
        println!("singular values: {:?}", s);

        let (mut u_keep, v_next, rank_new) = truncate_svd(&u, &s, &vt, cutoff, max_rank);

        if k > 0 {
            // (rank*2, r) -> (rank, 2, r)
            u_keep = u_keep.into_shape((rank, 2, rank_new))?;
        } else {
            // 最初は (2, r) を (1,2,r) に升目化して統一
            u_keep = u_keep.into_shape((1, 2, rank_new))?;
        }
        println!("tensor shape: {:?}\n", u_keep.dim());
        qtt.push(u_keep);
        yt = v_next.into_raw_vec().into(); // 次の右端
        rank = rank_new;
    }
    // 最後の (rank, 2)
    let last = yt.into_shape((rank, 2))?;
    // 末端は (rank,2,1) で持ってもよいが、後段の縮約のため 2D のまま扱う
    if depth < 5 {
        println!("depth: {}", depth - 1);
        println!("tensor shape: {:?}\n", last.dim());
    }

    // reconstruction
    // yr: (2, r1) を最初のコアから開始（内部では (i,j) 形にする）
    // Python の einsum と同じ書き方で進める
    let mut yr = qtt[0].clone().into_shape((2, qtt[0].dim().2))?.to_owned(); // (2, r1)
    for k in 1..(depth-1) {
        // yr(i,j) × qtt[k](j,k,l)
