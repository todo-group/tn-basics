use anyhow::Result;
use ndarray::{Array1, Array2, ArrayD, Ix2, Ix3, s};
use ndarray_einsum::einsum;
use ndarray_linalg::Norm;
use plotters::prelude::{BitMapBackend, IntoDrawingArea};
use tn_basics::{
    EasySVD, MapStrToAnyhowErr,
    plot::{plot_error, plot_target_vs_qtt},
};

// Conversion of one-dimensional function into QTT representation

fn main() -> Result<()> {
    let depth: usize = 4;
    let npoints: usize = 1 << depth;
    let cutoff: f64 = 1e-10;

    // target function
    let x = Array1::linspace(0.0, 1.0, npoints);
    let y = x.mapv(f64::exp);
    //let y = x.mapv(f64::sin);
    //let y = x.mapv(|x| ( -(((x-0.5)/0.1).powi(2)) ).exp() / (0.1 * std::f64::consts::PI.sqrt()) );

    if depth < 5 {
        println!("x = {x}");
        println!("y = {y}\n");
    }

    // QTT decomposition
    let mut yt = y.clone().into_shape_with_order((1, y.len()))?;
    let mut qtt: Vec<ArrayD<f64>> = Vec::with_capacity(depth);
    let mut rank: usize = 1;
    for k in 0..(depth - 1) {
        println!("depth: {k}");
        let v_view = yt.to_shape((rank * 2, yt.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd()?;
        println!("singular values: {s}");
        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice(s![.., 0..rank_new]);
        let s = s.slice(s![0..rank_new]);
        let vt = vt.slice(s![0..rank_new, ..]);
        let u = if k > 0 {
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_owned().into_dyn()
        };
        qtt.push(u);
        yt = Array2::from_diag(&s).dot(&vt);
        rank = rank_new;
    }
    println!("depth: {}", depth - 1);
    let yt = yt.into_shape_clone((rank, 2))?;
    qtt.push(yt.into_dyn());
    if depth < 5 {
        println!("qtt: [");
        for t in &qtt {
            println!("{t}");
        }
        println!("]");
    }

    // reconstruction
    let mut yr = qtt[0].clone().into_dimensionality::<Ix2>()?;
    for k in 1..(depth - 1) {
        let tmp = einsum("ij,jkl->ikl", &[&yr, &qtt[k]])
            .map_str_err()?
            .into_dimensionality::<Ix3>()?;
        let shape = tmp.dim();
        yr = tmp.into_shape_clone((shape.0 * shape.1, shape.2))?;
    }
    let yr = einsum("ij,jk->ik", &[&yr, &qtt[depth - 1]])
        .map_str_err()?
        .into_dimensionality::<Ix2>()?;
    let yr = yr.flatten();
    println!("error = {}", (&yr - &y).norm() / npoints as f64);

    if depth < 5 {
        println!("reconstructed y:\n{}", yr);
    }
    plot_target_vs_qtt(
        &BitMapBackend::new("target_vs_qtt.png", (800, 600)).into_drawing_area(),
        &x,
        &y,
        &yr,
    )?;
    println!("Saved plot: {}", "target_vs_qtt.png");
    plot_error(
        &BitMapBackend::new("error.png", (800, 600)).into_drawing_area(),
        &x,
        &y,
        &yr,
    )?;
    println!("Saved plot: {}", "error.png");

    Ok(())
}
